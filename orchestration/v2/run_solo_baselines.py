"""
FairWatch V2 - Production Solo Baselines Engine (orchestration/v2/run_solo_baselines.py)
Executes Stage 4.1 (E1): 4 personas x models x 1,008 core benchmark rows.
Evaluates agents with zero upstream context (T=0, guided JSON schema decoding).
Supports model-isolated parallel execution and aggregation into competence_index.csv & solo_logodds.parquet.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from orchestration.v2.vllm_client import VLLMClientV2
from orchestration.harness.decoding import POLICIES, PRIMARY, validate_sampling
from agents.base_agent import normalize_decision, PARSE_OK, PARSE_UNPARSEABLE, PARSE_ERROR, _sanitize_no_decimals
from agents.risk_manager_agent import RiskManagerAgent
from agents.regulatory_agent import RegulatoryAgent
from agents.data_science_agent import DataScienceAgent
from agents.consumer_advocate_agent import ConsumerAdvocateAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

PERSONA_CLASSES = [
    ("risk_manager", RiskManagerAgent),
    ("regulatory", RegulatoryAgent),
    ("data_science", DataScienceAgent),
    ("consumer_advocate", ConsumerAdvocateAgent),
]

MODEL_PARAM_COUNTS = {
    "llama3b": "3B",
    "llama8b": "8B",
    "llama70b": "70B",
    "qwen3b": "3B",
    "qwen7b": "7B",
    "qwen72b": "72B"
}

def load_seed_schedule(path: str) -> Dict[Tuple[str, int], int]:
    df = pd.read_csv(path)
    schedule = {}
    for _, row in df.iterrows():
        cell_id = str(row["twin_cell_id"]).strip()
        rep = int(row["replicate"])
        seed = int(row["planned_seed"])
        schedule[(cell_id, rep)] = seed
    return schedule

def compute_ece(df_sub: pd.DataFrame, n_bins: int = 10) -> float:
    valid = df_sub[df_sub["parse_status"] == PARSE_OK].copy()
    if len(valid) == 0:
        return 0.0

    confs = []
    accs = []
    for _, row in valid.iterrows():
        conf = float(row["confidence_probability"]) / 100.0
        confs.append(conf)
        match = 1.0 if row["approval_decision"] == row["pi_star"] else 0.0
        accs.append(match)

    confs = np.array(confs)
    accs = np.array(accs)
    n = len(confs)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (confs >= low) & (confs <= high if i == n_bins - 1 else confs < high)
        n_bin = np.sum(mask)
        if n_bin > 0:
            bin_acc = np.mean(accs[mask])
            bin_conf = np.mean(confs[mask])
            ece += (n_bin / n) * abs(bin_acc - bin_conf)

    return float(round(ece, 6))

def compute_balanced_accuracy(df_sub: pd.DataFrame) -> float:
    valid = df_sub[df_sub["parse_status"] == PARSE_OK]
    if len(valid) == 0:
        return 0.0

    positives = valid[valid["pi_star"] == "approve"]
    negatives = valid[valid["pi_star"] == "deny"]

    tpr = (positives["approval_decision"] == "approve").mean() if len(positives) > 0 else 0.0
    tnr = (negatives["approval_decision"] == "deny").mean() if len(negatives) > 0 else 0.0

    return float(round((tpr + tnr) / 2.0, 6))

def evaluate_task(
    model_alias: str,
    port: int,
    persona_key: str,
    agent_cls: Any,
    row: Dict[str, Any],
    seed: int,
    policy: Any,
    guided: bool
) -> Dict[str, Any]:
    client = VLLMClientV2(base_url=f"http://127.0.0.1:{port}/v1", model_alias=model_alias)
    client.set_seed(seed)
    client.set_guided(guided)
    agent = agent_cls(client)
    agent.config.temperature = policy.temperature
    agent.config.top_p = policy.top_p
    agent.config.max_tokens = policy.max_tokens

    prompt = str(row["prompt"])
    t0 = time.time()
    rec = agent.evaluate_loan_application(prompt)
    latency = round(time.time() - t0, 4)

    rec["execution_latency_sec"] = latency
    rec["sampling_policy_fingerprint"] = policy.fingerprint()
    rec["policy_id"] = policy.policy_id
    rec["guided"] = guided
    rec["request_seed"] = seed
    rec["planned_seed"] = seed
    rec["model"] = model_alias
    rec["persona_key"] = persona_key

    rec["prompt_id"] = str(row["prompt_id"])
    rec["twin_cell_id"] = str(row["twin_cell_id"])
    rec["source_case_id"] = str(row.get("source_case_id", ""))
    rec["applicant_name"] = str(row.get("name") or row.get("applicant_name", "")).strip()
    rec["canonical_tuple"] = str(row.get("canonical_tuple", ""))
    rec["credit_band"] = str(row.get("credit_band", ""))
    rec["lti_band"] = str(row.get("lti_band", ""))
    rec["pi_star"] = str(row.get("pi_star", ""))
    rec["borderline"] = bool(row.get("borderline", False))

    return _sanitize_no_decimals(rec)

def run_model_evaluations(
    model_alias: str,
    port: int,
    df_bm: pd.DataFrame,
    seed_schedule: Dict[Tuple[str, int], int],
    out_dir: Path,
    policy: Any,
    guided: bool,
    concurrency: int
):
    records_file = out_dir / f"solo_records_{model_alias}.jsonl"
    existing_keys = set()
    if records_file.exists():
        with open(records_file, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    item = json.loads(line)
                    existing_keys.add((item["persona_key"], item["prompt_id"]))
        LOG.info(f"[{model_alias}] Loaded {len(existing_keys)} existing records from {records_file}")

    tasks = []
    for _, row in df_bm.iterrows():
        cell_id = str(row["twin_cell_id"]).strip()
        prompt_id = str(row["prompt_id"]).strip()
        seed = seed_schedule.get((cell_id, 1), 42)

        for p_key, p_cls in PERSONA_CLASSES:
            if (p_key, prompt_id) in existing_keys:
                continue
            tasks.append((model_alias, port, p_key, p_cls, row.to_dict(), seed, policy, guided))

    LOG.info(f"[{model_alias}] {len(tasks)} tasks remaining (concurrency={concurrency})")
    if not tasks:
        return

    start_t = time.time()
    completed = 0
    with open(records_file, "a", encoding="utf-8") as fp_out:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(evaluate_task, *t): t for t in tasks
            }
            for fut in as_completed(futures):
                res = fut.result()
                fp_out.write(json.dumps(res) + "\n")
                fp_out.flush()
                completed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    elapsed = time.time() - start_t
                    rate = completed / elapsed if elapsed > 0 else 0
                    LOG.info(f"[{model_alias}] Completed {completed}/{len(tasks)} ({rate:.1f} evals/sec)")

def aggregate_and_export(out_dir: Path, benchmark_path: str, policy_name: str, guided: bool):
    all_files = sorted(out_dir.glob("solo_records_*.jsonl"))
    if not all_files:
        LOG.warning(f"No solo_records_*.jsonl found in {out_dir}")
        return

    merged_file = out_dir / "solo_records.jsonl"
    all_records = []
    with open(merged_file, "w", encoding="utf-8") as fp_merged:
        for f in all_files:
            LOG.info(f"Reading records from {f}...")
            with open(f, "r", encoding="utf-8") as fp_in:
                for line in fp_in:
                    if line.strip():
                        item = json.loads(line)
                        fp_merged.write(line.strip() + "\n")
                        all_records.append(item)

    LOG.info(f"Aggregated {len(all_records)} total records into {merged_file}")
    df_records = pd.DataFrame(all_records)

    # Competence index
    competence_rows = []
    for (model, persona), group in df_records.groupby(["model", "persona_key"]):
        total_evals = len(group)
        ok_evals = (group["parse_status"] == PARSE_OK).sum()
        err_evals = (group["parse_status"] == PARSE_ERROR).sum()
        unparse_evals = (group["parse_status"] == PARSE_UNPARSEABLE).sum()

        valid = group[group["parse_status"] == PARSE_OK]
        agreement_pi_star = (valid["approval_decision"] == valid["pi_star"]).mean() if len(valid) > 0 else 0.0
        bal_acc = compute_balanced_accuracy(group)
        ece = compute_ece(group)

        param_count = MODEL_PARAM_COUNTS.get(model, "unknown")
        competence_rows.append({
            "model": model,
            "persona": persona,
            "param_count": param_count,
            "parameter_count_is_capability_variable": False,
            "agreement_pi_star": round(float(agreement_pi_star), 6),
            "balanced_accuracy": bal_acc,
            "expected_calibration_error": ece,
            "n_evaluations": int(total_evals),
            "n_parse_ok": int(ok_evals),
            "n_parse_error": int(err_evals),
            "n_parse_unparseable": int(unparse_evals)
        })

    df_comp = pd.DataFrame(competence_rows)
    comp_file = out_dir / "competence_index.csv"
    df_comp.to_csv(comp_file, index=False)
    LOG.info(f"Wrote competence index to {comp_file}")

    # solo_logodds.parquet
    logodds_rows = []
    for _, r in df_records.iterrows():
        dec = r.get("approval_decision")
        conf_int = r.get("confidence_probability")
        conf = float(conf_int) / 100.0 if conf_int is not None else 0.5

        if dec == "approve":
            p_approve = conf
        elif dec == "deny":
            p_approve = 1.0 - conf
        else:
            p_approve = 0.5

        p_approve_clamped = max(0.001, min(0.999, p_approve))
        s_k = math.log(p_approve_clamped / (1.0 - p_approve_clamped))
        solo_margin = abs(s_k)

        logodds_rows.append({
            "model": str(r.get("model")),
            "persona": str(r.get("persona_key")),
            "prompt_id": str(r.get("prompt_id")),
            "twin_cell_id": str(r.get("twin_cell_id")),
            "source_case_id": str(r.get("source_case_id")),
            "applicant_name": str(r.get("applicant_name")),
            "approval_decision": dec,
            "confidence_probability": conf_int,
            "prob_approve": round(p_approve, 6),
            "s_k": round(s_k, 6),
            "solo_margin": round(solo_margin, 6),
            "pi_star": str(r.get("pi_star")),
            "borderline": bool(r.get("borderline", False)),
            "credit_band": str(r.get("credit_band")),
            "lti_band": str(r.get("lti_band"))
        })

    df_logodds = pd.DataFrame(logodds_rows)
    logodds_file = out_dir / "solo_logodds.parquet"
    df_logodds.to_parquet(logodds_file, index=False)
    LOG.info(f"Wrote {len(df_logodds)} records to {logodds_file}")

    manifest = {
        "benchmark": benchmark_path,
        "models": sorted(list(df_records["model"].unique())),
        "policy": policy_name,
        "guided": guided,
        "n_evaluations_total": len(df_records),
        "competence_index_path": str(comp_file),
        "solo_logodds_path": str(logodds_file),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sha256_records": hashlib.sha256(merged_file.read_bytes()).hexdigest(),
        "sha256_competence": hashlib.sha256(comp_file.read_bytes()).hexdigest(),
        "sha256_logodds": hashlib.sha256(logodds_file.read_bytes()).hexdigest()
    }
    manifest_file = out_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)
    LOG.info(f"Wrote manifest to {manifest_file}")

def main():
    parser = argparse.ArgumentParser(description="FairWatch V2 Solo Baselines (E1)")
    parser.add_argument("--benchmark", default="data/derived/core_benchmark_v2.csv")
    parser.add_argument("--models", default="", help="Comma-separated model:port pairs")
    parser.add_argument("--seed-schedule", default="docs/prereg/seed_schedule.csv")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--policy", default="primary", choices=list(POLICIES.keys()))
    parser.add_argument("--guided", default="true", choices=["true", "false"])
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    guided_bool = (args.guided.lower() == "true")

    if args.aggregate_only:
        aggregate_and_export(out_dir, args.benchmark, args.policy, guided_bool)
        return

    df_bm = pd.read_csv(args.benchmark)
    seed_schedule = load_seed_schedule(args.seed_schedule)

    model_specs = []
    for mp in args.models.split(","):
        mp = mp.strip()
        if not mp:
            continue
        parts = mp.split(":")
        m_name = parts[0].strip()
        m_port = int(parts[1].strip())
        model_specs.append((m_name, m_port))

    # Run evaluations per model
    for model_alias, port in model_specs:
        run_model_evaluations(
            model_alias=model_alias,
            port=port,
            df_bm=df_bm,
            seed_schedule=seed_schedule,
            out_dir=out_dir,
            policy=POLICIES[args.policy],
            guided=guided_bool,
            concurrency=args.concurrency
        )

    # Automatically aggregate if all finished
    aggregate_and_export(out_dir, args.benchmark, args.policy, guided_bool)

if __name__ == "__main__":
    main()
