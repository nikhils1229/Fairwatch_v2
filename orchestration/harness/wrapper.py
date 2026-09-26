"""
FairWatch V2 - AsyncAgentWrapper
Single authoritative agent wrapper eliminating prompt leakage, enforcing sampling policies,
and standardizing error taxonomies across all multi-agent topologies.
"""

from typing import Dict, Any, Optional
import time
import logging
import re
from agents.base_agent import BaseAgent, PARSE_OK, PARSE_UNPARSEABLE, PARSE_ERROR
from orchestration.harness.decoding import SamplingPolicy, PRIMARY, validate_sampling

LOG = logging.getLogger(__name__)

class AsyncAgentWrapper:
    """
    Unified agent invocation wrapper.
    Ensures every generation passes through validated sampling policies,
    tracks retry provenance, maps failure modes to strict error taxonomies,
    and preserves full raw outputs.
    """
    def __init__(
        self,
        agent: Any,
        policy: SamplingPolicy = PRIMARY,
        guided: bool = True,
        request_id: Optional[str] = None
    ):
        self.agent = agent
        self.policy = validate_sampling(policy)
        self.guided = guided
        self.request_id = request_id

    def execute(self, application_text: str, upstream_context: str = "") -> Dict[str, Any]:
        """
        Executes the wrapped agent on application data combined with upstream context.
        Enforces error taxonomy and provenance tracking.
        """
        prompt = application_text
        if upstream_context.strip():
            prompt = f"{application_text}\n\n[PRIOR ADVISOR DELIBERATIONS]:\n{upstream_context}"

        start_time = time.time()
        try:
            # If agent has custom evaluate_loan_application (BaseAgent derivative)
            if hasattr(self.agent, "evaluate_loan_application"):
                cfg = getattr(self.agent, "config", None)
                had_temp = hasattr(cfg, "temperature")
                orig_temp = getattr(cfg, "temperature", None)
                had_top_p = hasattr(cfg, "top_p")
                orig_top_p = getattr(cfg, "top_p", None)
                had_top_k = hasattr(cfg, "top_k")
                orig_top_k = getattr(cfg, "top_k", None)
                had_rep_pen = hasattr(cfg, "repetition_penalty")
                orig_rep_pen = getattr(cfg, "repetition_penalty", None)
                had_max = hasattr(cfg, "max_tokens")
                orig_max = getattr(cfg, "max_tokens", None)
                had_json = hasattr(cfg, "json_mode")
                orig_json = getattr(cfg, "json_mode", None)

                if cfg is not None:
                    if had_temp: cfg.temperature = self.policy.temperature
                    if had_top_p: cfg.top_p = self.policy.top_p
                    if had_top_k: cfg.top_k = self.policy.top_k
                    if had_rep_pen: cfg.repetition_penalty = self.policy.repetition_penalty
                    if had_max: cfg.max_tokens = self.policy.max_tokens
                    if had_json: cfg.json_mode = self.guided
                try:
                    record = self.agent.evaluate_loan_application(prompt)
                finally:
                    if cfg is not None:
                        if had_temp: cfg.temperature = orig_temp
                        if had_top_p: cfg.top_p = orig_top_p
                        if had_top_k: cfg.top_k = orig_top_k
                        if had_rep_pen: cfg.repetition_penalty = orig_rep_pen
                        if had_max: cfg.max_tokens = orig_max
                        if had_json: cfg.json_mode = orig_json
            else:
                # Direct callable or synthesize_decision
                record = self.agent(prompt)

            duration = time.time() - start_time
            if not isinstance(record, dict):
                record = {
                    "approval_decision": None,
                    "parse_status": PARSE_ERROR,
                    "parse_detail": f"Agent returned non-dict type {type(record)!r}",
                    "raw_response": str(record)
                }

            # Map parse error details to standard error taxonomy on production path
            if record.get("parse_status") == PARSE_ERROR and "error_class" not in record:
                detail_lower = str(record.get("parse_detail", "")).lower()
                if "timeout" in detail_lower:
                    record["error_class"] = "timeout"
                elif any(term in detail_lower for term in ["500", "502", "503", "504", "http 5", "server error"]):
                    record["error_class"] = "http_5xx"
                elif re.search(r'\b(context[_\s-]?length\w*|maximum context|token\s*limit|tokens?\s+exceed\w*|prompt\s+(?:is\s+)?too\s+long|exceed\w*\s+(?:\d+\s+)?tokens?)\b', detail_lower):
                    record["error_class"] = "context_overflow"
                else:
                    record["error_class"] = "client_exception"

            record["sampling_policy_fingerprint"] = self.policy.fingerprint()
            record["policy_id"] = self.policy.policy_id
            record["guided_decoding"] = self.guided
            record["execution_latency_sec"] = round(duration, 4)
            if self.request_id:
                record["request_id"] = self.request_id

            return record

        except Exception as exc:
            duration = time.time() - start_time
            exc_str = str(exc).lower()

            # Status code detection from exception or response object
            status_code = getattr(exc, "status_code", None) or getattr(exc, "status", None)
            if hasattr(exc, "response") and exc.response is not None:
                status_code = getattr(exc.response, "status_code", status_code)

            if isinstance(exc, (TimeoutError,)):
                err_class = "timeout"
            elif "timeout" in exc_str:
                err_class = "timeout"
            elif status_code is not None and 500 <= int(status_code) <= 599:
                err_class = "http_5xx"
            elif re.search(r'\b50[0-4]\b', exc_str) and ("server" in exc_str or "http" in exc_str):
                err_class = "http_5xx"
            elif re.search(r'\b(context[_\s-]?length\w*|maximum context|token\s*limit|tokens?\s+exceed\w*|prompt\s+(?:is\s+)?too\s+long|exceed\w*\s+(?:\d+\s+)?tokens?)\b', exc_str):
                err_class = "context_overflow"
            else:
                err_class = "client_exception"

            LOG.error(f"Wrapper execution failed ({err_class}): {exc}")
            return {
                "agent_name": getattr(self.agent, "agent_name", "UnknownAgent"),
                "approval_decision": None,
                "approval_type": None,
                "interest_rate": None,
                "confidence_probability": None,
                "confidence_level": None,
                "parse_status": PARSE_ERROR,
                "parse_detail": f"{err_class}: {str(exc)}",
                "error_class": err_class,
                "raw_response": "",
                "raw_excerpt": "",
                "sampling_policy_fingerprint": self.policy.fingerprint(),
                "policy_id": self.policy.policy_id,
                "guided_decoding": self.guided,
                "execution_latency_sec": round(duration, 4),
                "request_id": self.request_id
            }
