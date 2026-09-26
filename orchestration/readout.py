"""
FairWatch V2 - Readout Aggregation Protocols (R1, R2, R3)
Implements deterministic majority voting (R1), LLM synthesis judging (R2), and terminal agent readout (R3).
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from agents.base_agent import normalize_decision, PARSE_OK, PARSE_UNPARSEABLE, PARSE_ERROR
from orchestration.harness.decoding import SamplingPolicy, PRIMARY

def R1_majority(records_or_decisions: List[Any], tie: str = "deny") -> Dict[str, Any]:
    """
    R1 Majority Voting Readout:
    Counts valid decisions ('approve', 'deny') across participating agents.
    Ties broken according to conservative credit baseline ('deny').
    Strictly invariant to input permutation (order-independent).
    """
    decisions = []
    for item in records_or_decisions:
        if isinstance(item, dict):
            dec = item.get("approval_decision")
        else:
            dec = item
        norm = normalize_decision(dec)
        if norm in ("approve", "deny"):
            decisions.append(norm)

    counts = Counter(decisions)
    n_approve = counts.get("approve", 0)
    n_deny = counts.get("deny", 0)
    total_valid = n_approve + n_deny

    if total_valid == 0:
        final_decision = None
        status = PARSE_UNPARSEABLE
    elif n_approve > n_deny:
        final_decision = "approve"
        status = PARSE_OK
    elif n_deny > n_approve:
        final_decision = "deny"
        status = PARSE_OK
    else:
        # Exact tie
        final_decision = tie
        status = PARSE_OK

    return {
        "readout_protocol": "R1_majority",
        "approval_decision": final_decision,
        "vote_counts": {"approve": n_approve, "deny": n_deny},
        "total_valid_votes": total_valid,
        "parse_status": status,
        "tie_broken": (n_approve == n_deny and total_valid > 0)
    }


def R2_judge(
    agent_outputs: List[Dict[str, Any]],
    client: Any,
    policy: SamplingPolicy = PRIMARY,
    rho_order: Optional[List[str]] = None,
    label_mode: str = "named"
) -> Dict[str, Any]:
    """
    R2 Synthesis Judge Readout:
    Takes multi-agent deliberations, orders them according to rho_order (or default),
    and queries an independent judge model to synthesize a final verdict.
    """
    ordered_outputs = list(agent_outputs)
    if rho_order:
        def get_order_idx(rec):
            name = rec.get("agent_name", "")
            try:
                return rho_order.index(name)
            except ValueError:
                return len(rho_order)
        ordered_outputs.sort(key=get_order_idx)

    if label_mode not in ("named", "anonymous", "blinded"):
        raise ValueError(f"Invalid label_mode: '{label_mode}'. Must be one of: 'named', 'anonymous', 'blinded'")

    formatted_recs = []
    for i, rec in enumerate(ordered_outputs):
        if label_mode in ("anonymous", "blinded"):
            name = f"Advisor_{chr(65+i)}"
        else:
            name = rec.get("agent_name", f"Agent_{i+1}")
        dec = rec.get("approval_decision", "N/A")
        rate = rec.get("interest_rate", "N/A")
        reasoning = rec.get("reasoning", {})
        rationale = reasoning.get("approval_decision_reason", "N/A") if isinstance(reasoning, dict) else "N/A"
        formatted_recs.append(f"[{name}] Recommendation: {dec} | Rate: {rate}%\nRationale: {rationale}")

    judge_prompt = (
        "You are the Senior Presiding Underwriter. Review the following peer advisor deliberations:\n\n"
        + "\n\n".join(formatted_recs)
        + "\n\nSynthesize these inputs into a final credit decision. Return ONLY valid JSON:\n"
        '{"approval_decision": "approve or deny", "interest_rate": 8.5, "synthesis_rationale": "Explanation"}'
    )

    try:
        from orchestration.clients.ollama_client import GenerationConfig
        cfg = GenerationConfig(
            temperature=policy.temperature,
            max_tokens=policy.max_tokens,
            top_p=policy.top_p,
            json_mode=True,
            top_k=policy.top_k,
            repetition_penalty=policy.repetition_penalty
        )
        if hasattr(client, "generate"):
            import inspect
            sig = inspect.signature(client.generate)
            if "config" in sig.parameters:
                res = client.generate(prompt=judge_prompt, system_prompt=None, config=cfg)
            else:
                res = client.generate(
                    prompt=judge_prompt,
                    temperature=policy.temperature,
                    top_p=policy.top_p,
                    max_tokens=policy.max_tokens
                )
        else:
            res = client(judge_prompt)

        if hasattr(res, "success") and not res.success:
            return {
                "readout_protocol": "R2_judge",
                "approval_decision": None,
                "error": getattr(res, "error_message", "Generation failed"),
                "parse_status": PARSE_ERROR,
                "sampling_policy_fingerprint": policy.fingerprint(),
                "label_mode": label_mode,
                "applied_rho_order": rho_order
            }
        text = res.text if hasattr(res, "text") else str(res)
    except Exception as e:
        return {
            "readout_protocol": "R2_judge",
            "approval_decision": None,
            "error": str(e),
            "parse_status": PARSE_ERROR,
            "sampling_policy_fingerprint": policy.fingerprint(),
            "label_mode": label_mode,
            "applied_rho_order": rho_order
        }

    try:
        import json
        from agents.base_agent import BaseAgent
        cleaned_json_str = BaseAgent._clean_json(None, text)
        parsed = json.loads(cleaned_json_str)
        if not isinstance(parsed, dict):
            return {
                "readout_protocol": "R2_judge",
                "approval_decision": None,
                "raw_response": text,
                "parse_status": PARSE_UNPARSEABLE,
                "sampling_policy_fingerprint": policy.fingerprint(),
                "label_mode": label_mode,
                "applied_rho_order": rho_order
            }
        decision = normalize_decision(parsed.get("approval_decision"))
        return {
            "readout_protocol": "R2_judge",
            "approval_decision": decision,
            "interest_rate": parsed.get("interest_rate"),
            "raw_response": text,
            "parse_status": PARSE_OK if decision else PARSE_UNPARSEABLE,
            "sampling_policy_fingerprint": policy.fingerprint(),
            "label_mode": label_mode,
            "applied_rho_order": rho_order
        }
    except Exception as e:
        return {
            "readout_protocol": "R2_judge",
            "approval_decision": None,
            "raw_response": text,
            "parse_detail": f"Unparseable JSON: {str(e)}",
            "parse_status": PARSE_UNPARSEABLE,
            "sampling_policy_fingerprint": policy.fingerprint(),
            "label_mode": label_mode,
            "applied_rho_order": rho_order
        }


def R3_terminal(
    agent_outputs: List[Dict[str, Any]],
    terminal_role: str = "Business Decision"
) -> Dict[str, Any]:
    """
    R3 Terminal Agent Readout:
    Directly extracts the decision emitted by the designated terminal synthesizer role.
    Strict exact matching on terminal role name.
    """
    target = terminal_role.strip().lower()
    for rec in reversed(agent_outputs):
        name = str(rec.get("agent_name", "")).strip().lower()
        if name == target:
            dec = normalize_decision(rec.get("approval_decision"))
            return {
                "readout_protocol": "R3_terminal",
                "terminal_role": terminal_role,
                "approval_decision": dec,
                "interest_rate": rec.get("interest_rate"),
                "agent_influence": rec.get("agent_influence"),
                "parse_status": rec.get("parse_status", PARSE_OK if dec else PARSE_UNPARSEABLE)
            }

    # Fallback to very last record if designated role not found
    if agent_outputs:
        last = agent_outputs[-1]
        dec = normalize_decision(last.get("approval_decision"))
        return {
            "readout_protocol": "R3_terminal",
            "terminal_role": "last_record_fallback",
            "approval_decision": dec,
            "interest_rate": last.get("interest_rate"),
            "agent_influence": last.get("agent_influence"),
            "parse_status": last.get("parse_status", PARSE_OK if dec else PARSE_UNPARSEABLE)
        }

    return {
        "readout_protocol": "R3_terminal",
        "terminal_role": terminal_role,
        "approval_decision": None,
        "parse_status": PARSE_UNPARSEABLE
    }
