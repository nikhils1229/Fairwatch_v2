"""
FairWatch V2 - MultiAgentRunner
Executes credit underwriting deliberations across Sequential (SEQ), Hierarchical (HIER),
Peer-Review (PEER), and Parallel (PAR) network topologies.
"""

from typing import List, Dict, Any, Optional
import time
import re
from orchestration.harness.wrapper import AsyncAgentWrapper
from orchestration.harness.transmission import phi_full, phi_evidence, phi_verdict, sanitize_evidence_text
from orchestration.harness.readout import R1_majority, R2_judge, R3_terminal
from orchestration.harness.decoding import SamplingPolicy, PRIMARY
from agents.base_agent import normalize_decision, PARSE_ERROR

class MultiAgentRunner:
    def __init__(
        self,
        agents: List[Any],
        synthesizer: Optional[Any] = None,
        policy: SamplingPolicy = PRIMARY,
        phi_fn = phi_full
    ):
        self.agents = agents
        self.synthesizer = synthesizer
        self.policy = policy
        self.phi_fn = phi_fn

    def run_seq(self, application_data: str, order_sigma: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Sequential Topology (SEQ):
        Agents execute strictly in order sigma = (sigma_1, sigma_2, ...).
        Downstream agents receive transmission phi(prior_agents).
        Emits upstream_verdicts for cascade analysis (Fable Audit Item 1).
        """
        start_t = time.time()
        n = len(self.agents)
        if order_sigma is not None:
            if sorted(order_sigma) != list(range(n)):
                raise ValueError(f"order_sigma {order_sigma} is not a valid permutation of range({n})")
            sigma = order_sigma
        else:
            sigma = list(range(n))

        ordered_agents = [self.agents[i] for i in sigma]
        
        agent_records = []
        deliberation_history = []
        prior_verdicts = []

        for idx, agent in enumerate(ordered_agents):
            wrapper = AsyncAgentWrapper(agent, policy=self.policy)
            context_str = "\n\n".join(deliberation_history)
            rec = wrapper.execute(application_data, upstream_context=context_str)
            rec["seq_position"] = idx + 1
            rec["upstream_verdicts"] = list(prior_verdicts) # Exact upstream decision sequence
            
            agent_records.append(rec)
            prior_verdicts.append(normalize_decision(rec.get("approval_decision")))
            
            # Format transmission for downstream
            phi_text, _ = self.phi_fn(rec)
            deliberation_history.append(phi_text)

        readout = R1_majority(agent_records)
        return {
            "topology": "SEQ",
            "order_sigma": sigma,
            "agent_records": agent_records,
            "readout": readout,
            "final_decision": readout.get("approval_decision"),
            "tie_broken": readout.get("tie_broken", False),
            "duration_sec": round(time.time() - start_t, 4)
        }

    def run_hier(self, application_data: str) -> Dict[str, Any]:
        """
        Hierarchical Topology (HIER):
        Domain specialist agents run in parallel without cross-talk.
        Synthesizer (BusinessDecisionAgent) evaluates recommendations formatted via phi_fn.
        """
        start_t = time.time()
        agent_records = []
        for agent in self.agents:
            wrapper = AsyncAgentWrapper(agent, policy=self.policy)
            rec = wrapper.execute(application_data)
            agent_records.append(rec)

        if self.synthesizer:
            transmitted_records = []
            for r in agent_records:
                if self.phi_fn == phi_evidence:
                    masked_r = dict(r)
                    masked_r["approval_decision"] = None
                    masked_r["approval_type"] = None
                    masked_r["interest_rate"] = None
                    masked_r["confidence_probability"] = None
                    masked_r["confidence_level"] = None
                    masked_r["raw_response"] = ""
                    masked_r["raw_excerpt"] = ""
                    if isinstance(masked_r.get("reasoning"), dict):
                        masked_reasoning = {k: sanitize_evidence_text(str(v)) for k, v in masked_r["reasoning"].items()}
                        masked_r["reasoning"] = masked_reasoning
                    transmitted_records.append(masked_r)
                elif self.phi_fn == phi_verdict:
                    # Strip narrative reasoning and raw outputs
                    masked_r = dict(r)
                    masked_r["reasoning"] = {}
                    masked_r["raw_response"] = ""
                    masked_r["raw_excerpt"] = ""
                    transmitted_records.append(masked_r)
                else:
                    transmitted_records.append(r)

            # Wrapped synthesizer execution to guarantee sampling policy provenance and error handling
            synth_wrapper = AsyncAgentWrapper(self.synthesizer, policy=self.policy)
            try:
                if hasattr(self.synthesizer, "synthesize_decision"):
                    synth_rec = self.synthesizer.synthesize_decision(application_data, transmitted_records)
                    synth_rec["sampling_policy_fingerprint"] = self.policy.fingerprint()
                    synth_rec["policy_id"] = self.policy.policy_id
                else:
                    synth_rec = synth_wrapper.execute(application_data)
            except Exception as e:
                exc_str = str(e).lower()
                if isinstance(e, TimeoutError) or "timeout" in exc_str:
                    err_class = "timeout"
                elif any(term in exc_str for term in ["500", "502", "503", "504", "http 5"]):
                    err_class = "http_5xx"
                elif re.search(r'\b(context[_\s-]?length|maximum context|token limit|tokens? exceeded|prompt (?:is )?too long)\b', exc_str):
                    err_class = "context_overflow"
                else:
                    err_class = "client_exception"

                synth_rec = {
                    "agent_name": getattr(self.synthesizer, "agent_name", "Synthesizer"),
                    "approval_decision": None,
                    "approval_type": None,
                    "interest_rate": None,
                    "parse_status": PARSE_ERROR,
                    "parse_detail": f"Synthesizer execution error ({err_class}): {str(e)}",
                    "error_class": err_class,
                    "raw_response": "",
                    "raw_excerpt": "",
                    "sampling_policy_fingerprint": self.policy.fingerprint(),
                    "policy_id": self.policy.policy_id
                }
            readout = R3_terminal([synth_rec])
        else:
            readout = R1_majority(agent_records)

        return {
            "topology": "HIER",
            "agent_records": agent_records,
            "synthesis_record": synth_rec if self.synthesizer else None,
            "readout": readout,
            "final_decision": readout.get("approval_decision"),
            "tie_broken": readout.get("tie_broken", False),
            "duration_sec": round(time.time() - start_t, 4)
        }

    def run_par(self, application_data: str) -> Dict[str, Any]:
        """
        Parallel Topology (PAR):
        Agents evaluate the loan application in strict mutual isolation.
        Readout protocol (R1 majority) determines consensus.
        """
        start_t = time.time()
        agent_records = []
        for agent in self.agents:
            wrapper = AsyncAgentWrapper(agent, policy=self.policy)
            rec = wrapper.execute(application_data)
            agent_records.append(rec)

        readout = R1_majority(agent_records)
        return {
            "topology": "PAR",
            "agent_records": agent_records,
            "readout": readout,
            "final_decision": readout.get("approval_decision"),
            "tie_broken": readout.get("tie_broken", False),
            "duration_sec": round(time.time() - start_t, 4)
        }

    def run_peer(self, application_data: str, n_rounds: int = 1) -> Dict[str, Any]:
        """
        Peer Review Topology (PEER):
        Round 1: Parallel initial recommendations.
        Round 2..n: Cross-examination of peer recommendations before final aggregation.
        """
        start_t = time.time()
        n_rounds = max(1, n_rounds)
        round1_records = []
        for agent in self.agents:
            wrapper = AsyncAgentWrapper(agent, policy=self.policy)
            rec = wrapper.execute(application_data)
            round1_records.append(rec)

        current_records = round1_records
        all_rounds = [round1_records]

        for r in range(2, max(2, n_rounds + 1)):
            round_r_records = []
            for i, agent in enumerate(self.agents):
                peers = [rec for j, rec in enumerate(current_records) if j != i]
                peer_context = "\n\n".join([self.phi_fn(p)[0] for p in peers])
                wrapper = AsyncAgentWrapper(agent, policy=self.policy)
                rec = wrapper.execute(application_data, upstream_context=f"PEER REVIEW ROUND {r-1}:\n{peer_context}")
                round_r_records.append(rec)
            all_rounds.append(round_r_records)
            current_records = round_r_records

        final_readout = R1_majority(current_records)
        return {
            "topology": "PEER",
            "n_rounds": n_rounds,
            "rounds_history": all_rounds,
            "agent_records": current_records,
            "readout": final_readout,
            "final_decision": final_readout.get("approval_decision"),
            "tie_broken": final_readout.get("tie_broken", False),
            "duration_sec": round(time.time() - start_t, 4)
        }
