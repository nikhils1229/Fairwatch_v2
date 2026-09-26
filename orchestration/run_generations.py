"""
FairWatch V2 - Multi-Agent Generation Engine (orchestration/v2/run_generations.py)
Generates deliberation message-sets across PAR, SEQ, HIER, and PEER network topologies.
Enforces seed schedules, sampling policies, guided JSON decoding, and zero-truncation transmission.
Supports concurrent workers, arbitrary order permutations (including full-24), and checkpoint resumption.
"""

import argparse
import hashlib
import itertools
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

from orchestration.v2.vllm_client import VLLMClientV2
from orchestration.harness.decoding import POLICIES, PRIMARY, validate_sampling
from orchestration.harness.transmission import phi_full
from agents.base_agent import normalize_decision, PARSE_OK, PARSE_UNPARSEABLE, PARSE_ERROR, _sanitize_no_decimals
from agents.risk_manager_agent import RiskManagerAgent
from agents.regulatory_agent import RegulatoryAgent
from agents.data_science_agent import DataScienceAgent
from agents.consumer_advocate_agent import ConsumerAdvocateAgent
from agents.business_decision_agent import BusinessDecisionAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

def compute_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def load_seed_schedule(path: str) -> Dict[tuple, int]:
    df = pd.read_csv(path)
    schedule = {}
    for _, row in df.iterrows():
        cell_id = str(row["twin_cell_id"]).strip()
        rep = int(row["replicate"])
        seed = int(row["planned_seed"])
        schedule[(cell_id, rep)] = seed
    return schedule

def create_agents(client: VLLMClientV2):
    return [
        RiskManagerAgent(client),
        RegulatoryAgent(client),
        DataScienceAgent(client),
        ConsumerAdvocateAgent(client)
    ]

def run_case_par(agents: List[Any], prompt: str, seed: int, policy, guided: bool) -> List[Dict[str, Any]]:
    records = []
    for agent in agents:
        agent.client.set_seed(seed)
        agent.client.set_guided(guided)
        agent.config.temperature = policy.temperature
        agent.config.top_p = policy.top_p
        agent.config.max_tokens = policy.max_tokens

        t0 = time.time()
        rec = agent.evaluate_loan_application(prompt)
        rec["execution_latency_sec"] = round(time.time() - t0, 4)
        rec["sampling_policy_fingerprint"] = policy.fingerprint()
        rec["policy_id"] = policy.policy_id
        rec["guided"] = guided
        rec["request_seed"] = seed
        rec["planned_seed"] = seed
        records.append(rec)
    return records

def run_case_seq(agents: List[Any], prompt: str, seed: int, policy, guided: bool, order_sigma: List[int]) -> List[Dict[str, Any]]:
    ordered_agents = [agents[i] for i in order_sigma]
    records = []
    deliberation_history = []
    prior_verdicts = []

    for idx, agent in enumerate(ordered_agents):
        agent.client.set_seed(seed)
        agent.client.set_guided(guided)
        agent.config.temperature = policy.temperature
        agent.config.top_p = policy.top_p
        agent.config.max_tokens = policy.max_tokens

        context_str = "\n\n".join(deliberation_history)
        eval_prompt = prompt if not context_str else f"{prompt}\n\n[PRIOR ADVISOR DELIBERATIONS]:\n{context_str}"

        t0 = time.time()
        rec = agent.evaluate_loan_application(eval_prompt)
        rec["execution_latency_sec"] = round(time.time() - t0, 4)
        rec["seq_position"] = idx + 1
        rec["upstream_verdicts"] = list(prior_verdicts)
        rec["sampling_policy_fingerprint"] = policy.fingerprint()
        rec["policy_id"] = policy.policy_id
        rec["guided"] = guided
        rec["request_seed"] = seed
        rec["planned_seed"] = seed

        records.append(rec)
        prior_verdicts.append(normalize_decision(rec.get("approval_decision")))

        phi_text, _ = phi_full(rec)
        deliberation_history.append(phi_text)

    return records

def run_case_hier(agents: List[Any], manager: Any, prompt: str, seed: int, policy, guided: bool) -> tuple:
    domain_records = run_case_par(agents, prompt, seed, policy, guided)
    manager.client.set_seed(seed)
    manager.client.set_guided(guided)

    t0 = time.time()
    synth_rec = manager.synthesize_decision(prompt, domain_records)
    synth_rec["execution_latency_sec"] = round(time.time() - t0, 4)
    synth_rec["sampling_policy_fingerprint"] = policy.fingerprint()
    synth_rec["policy_id"] = policy.policy_id
    synth_rec["guided"] = guided
    synth_rec["request_seed"] = seed
    synth_rec["planned_seed"] = seed

    return domain_records, synth_rec

def run_case_peer(agents: List[Any], prompt: str, seed: int, policy, guided: bool, frozen_round1: Optional[List[Dict[str, Any]]]) -> tuple:
    if frozen_round1 is not None:
        round1_records = list(frozen_round1)
        regenerated_r1 = 0
    else:
        round1_records = run_case_par(agents, prompt, seed, policy, guided)
        regenerated_r1 = 1

    round2_records = []
    for reviewer_idx, reviewer_agent in enumerate(agents):
        target_idx = (reviewer_idx + 1) % len(agents)
        target_eval = round1_records[target_idx]

        phi_text, _ = phi_full(target_eval)
        peer_prompt = (
            f"{prompt}\n\n"
            f"[PEER ADVISOR EVALUATION UNDER REVIEW]:\n{phi_text}\n\n"
            f"Please review the applicant and peer recommendation, and provide your final independent loan evaluation."
        )

        reviewer_agent.client.set_seed(seed)
        reviewer_agent.client.set_guided(guided)
        reviewer_agent.config.temperature = policy.temperature
        reviewer_agent.config.top_p = policy.top_p
        reviewer_agent.config.max_tokens = policy.max_tokens

        t0 = time.time()
        r2_rec = reviewer_agent.evaluate_loan_application(peer_prompt)
        r2_rec["execution_latency_sec"] = round(time.time() - t0, 4)
        r2_rec["peer_round"] = 2
        r2_rec["reviewed_agent"] = target_eval.get("agent_name")
        r2_rec["sampling_policy_fingerprint"] = policy.fingerprint()
        r2_rec["policy_id"] = policy.policy_id
        r2_rec["guided"] = guided
        r2_rec["request_seed"] = seed
        r2_rec["planned_seed"] = seed
        round2_records.append(r2_rec)

    return round1_records, round2_records, regenerated_r1

def process_case_task(
    row_dict: Dict[str, Any],
    seed: int,
    topology: str,
    base_url: str,
    model_alias: str,
    policy: Any,
    guided_bool: bool,
    frozen_round1: Optional[List[Dict[str, Any]]],
    order_sigma: Optional[List[int]] = None
) -> Dict[str, Any]:
    client = VLLMClientV2(base_url=base_url, model_alias=model_alias)
    client.set_guided(guided_bool)
    agents = create_agents(client)
    manager = BusinessDecisionAgent(client) if topology == "HIER" else None

    prompt = str(row_dict["prompt"])
    cell_id = str(row_dict["twin_cell_id"]).strip()
    name = str(row_dict.get("name") or row_dict.get("applicant_name")).strip()
    prompt_id = str(row_dict["prompt_id"])
    canon_tuple = str(row_dict.get("canonical_tuple", ""))

    t0 = time.time()
    if topology == "PAR":
        agent_recs = run_case_par(agents, prompt, seed, policy, guided_bool)
        arity = len(agent_recs)
        all_recs = agent_recs
        mgr_rec = None
        r1_recs = None
        r2_recs = None
        sigma = [0, 1, 2, 3]
    elif topology == "SEQ":
        sigma = order_sigma if order_sigma is not None else [0, 1, 2, 3]
        agent_recs = run_case_seq(agents, prompt, seed, policy, guided_bool, sigma)
        arity = len(agent_recs)
        all_recs = agent_recs
        mgr_rec = None
        r1_recs = None
        r2_recs = None
    elif topology == "HIER":
        domain_recs, mgr_rec = run_case_hier(agents, manager, prompt, seed, policy, guided_bool)
        arity = len(domain_recs) + 1
        all_recs = domain_recs
        r1_recs = None
        r2_recs = None
        sigma = [0, 1, 2, 3]
    elif topology == "PEER":
        r1_recs, r2_recs, regen = run_case_peer(agents, prompt, seed, policy, guided_bool, frozen_round1)
        arity = len(r1_recs) + len(r2_recs)
        all_recs = r2_recs
        mgr_rec = None
        sigma = [0, 1, 2, 3]

    case_dur = round(time.time() - t0, 4)

    statuses = [r.get("parse_status", PARSE_OK) for r in all_recs]
    if mgr_rec:
        statuses.append(mgr_rec.get("parse_status", PARSE_OK))
    agg_status = PARSE_ERROR if PARSE_ERROR in statuses else PARSE_UNPARSEABLE if PARSE_UNPARSEABLE in statuses else PARSE_OK

    out_record = {
        "twin_cell_id": cell_id,
        "applicant_name": name,
        "name": name,
        "prompt_id": prompt_id,
        "canonical_tuple": canon_tuple,
        "topology": topology,
        "order_sigma": sigma,
        "sigma_id": f"sigma_{'_'.join(map(str, sigma))}",
        "model_alias": model_alias,
        "policy_id": policy.policy_id,
        "sampling_policy_fingerprint": policy.fingerprint(),
        "guided": guided_bool,
        "request_seed": seed,
        "planned_seed": seed,
        "replicate": 1,
        "chain_arity": arity,
        "agent_records": all_recs,
        "manager_record": mgr_rec,
        "round1_records": r1_recs,
        "round2_records": r2_recs,
        "parse_status": agg_status,
        "message_set_hash": compute_hash(all_recs),
        "duration_sec": case_dur
    }
    return _sanitize_no_decimals(out_record)

def main():
    parser = argparse.ArgumentParser(description="FairWatch V2 Run Generations")
    parser.add_argument("--benchmark", type=str, required=True, help="Path to benchmark CSV")
    parser.add_argument("--cells", type=str, required=True, help="Path to cells text file")
    parser.add_argument("--topology", type=str, required=True, choices=["PAR", "SEQ", "HIER", "PEER"])
    parser.add_argument("--orders", type=str, default="canonical", choices=["canonical", "full24", "reduced12"])
    parser.add_argument("--model", type=str, default="llama8b")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--policy", type=str, default="primary")
    parser.add_argument("--guided", type=str, default="true")
    parser.add_argument("--seed-schedule", type=str, default="docs/prereg/seed_schedule.csv")
    parser.add_argument("--reuse-frozen", type=str, default=None, help="Path to frozen PAR runs to reuse round 1 for PEER")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent workers")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    guided_bool = args.guided.lower() in ("true", "1", "yes")
    policy = POLICIES.get(args.policy, PRIMARY)
    base_url = f"http://127.0.0.1:{args.port}/v1"

    df_bm = pd.read_csv(args.benchmark)
    with open(args.cells, "r", encoding="utf-8") as fp:
        target_cells = set(line.strip() for line in fp if line.strip())

    sub_df = df_bm[df_bm["twin_cell_id"].isin(target_cells)].copy()
    LOG.info(f"Filtered benchmark to {len(sub_df)} cases across {len(target_cells)} target cells for topology {args.topology}")

    seed_schedule = load_seed_schedule(args.seed_schedule)

    frozen_par_map = {}
    if args.topology == "PEER" and args.reuse_frozen:
        reuse_p = Path(args.reuse_frozen)
        for f in reuse_p.glob("*.jsonl"):
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    if line.strip():
                        item = json.loads(line)
                        k = (item["twin_cell_id"], item.get("applicant_name") or item.get("name"))
                        frozen_par_map[k] = item.get("agent_records", [])
        LOG.info(f"Loaded {len(frozen_par_map)} frozen PAR cases for PEER round 1 reuse")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "message_sets.jsonl"

    existing_keys = set()
    if out_file.exists():
        with open(out_file, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    item = json.loads(line)
                    sig = tuple(item.get("order_sigma", [0, 1, 2, 3]))
                    existing_keys.add((item.get("prompt_id"), sig))
        LOG.info(f"Found {len(existing_keys)} existing cases in {out_file}")

    # Determine permutations
    if args.topology == "SEQ" and args.orders == "full24":
        sigmas = list(itertools.permutations([0, 1, 2, 3]))
    elif args.topology == "SEQ" and args.orders == "reduced12":
        od_df = pd.read_csv("docs/prereg/order_design.csv")
        sigmas = [eval(row["order_tuple"]) for _, row in od_df.iterrows()]
        LOG.info(f"Loaded {len(sigmas)} balanced permutations from docs/prereg/order_design.csv")
    else:
        sigmas = [[0, 1, 2, 3]]

    tasks = []
    for idx, row in sub_df.iterrows():
        pid = str(row["prompt_id"])
        cell_id = str(row["twin_cell_id"]).strip()
        seed = seed_schedule.get((cell_id, 1), 42)
        name = str(row.get("name") or row.get("applicant_name")).strip()
        frozen_r1 = frozen_par_map.get((cell_id, name))

        for sig in sigmas:
            sig_tuple = tuple(sig)
            if (pid, sig_tuple) in existing_keys:
                continue
            tasks.append((row.to_dict(), seed, args.topology, base_url, args.model, policy, guided_bool, frozen_r1, list(sig)))

    LOG.info(f"Executing {len(tasks)} tasks for {args.topology} ({len(sigmas)} order(s)) with concurrency={args.concurrency}")
    if not tasks:
        LOG.info(f"All cases already completed in {out_file}")
        return

    start_total_t = time.time()
    completed = 0
    with open(out_file, "a", encoding="utf-8") as fp_out:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(process_case_task, *t): t for t in tasks
            }
            for fut in as_completed(futures):
                res = fut.result()
                fp_out.write(json.dumps(res) + "\n")
                fp_out.flush()
                completed += 1
                if completed % 24 == 0 or completed == len(tasks):
                    elapsed = time.time() - start_total_t
                    rate = completed / elapsed if elapsed > 0 else 0
                    LOG.info(f"[{args.topology}] Completed {completed}/{len(tasks)} ({rate:.2f} cases/sec, {completed*100//len(tasks)}%)")

    LOG.info(f"[{args.topology}] Finished all {completed} cases in {round(time.time()-start_total_t, 1)}s. Saved to {out_file}")

if __name__ == "__main__":
    main()
