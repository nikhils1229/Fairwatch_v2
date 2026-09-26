#!/usr/bin/env python3
"""
Evaluation and Statistical Verification of D2b Precision/Decoding Robustness:
Evaluates stochastic decoding on Qwen-72B across T in {0.20, 0.70} and 3 random seeds.
Compares final decision against upstream majority verdict (Positions 1-3).
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPSTREAM_FILE = BASE_DIR / "results/v2/production/readout_R1/SEQ_canonical_qwen72b.jsonl"
D2_DIR = BASE_DIR / "results/v2/ablations/D2_precision_decoding"

def main():
    print("=" * 80)
    print("EVALUATING D2b STOCHASTIC DECODING ROBUSTNESS (QWEN-72B)")
    print("=" * 80)

    # Load upstream majority from SEQ_canonical_qwen72b.jsonl
    upstream_maj = {}
    with open(UPSTREAM_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec.get("prompt_id")
            agent_records = rec.get("agent_records", [])
            first_three = agent_records[:3]
            app_count = sum(1 for r in first_three if str(r.get("approval_decision", "")).strip().lower() in ("approve", "approved"))
            maj = "approve" if app_count >= 2 else "deny"
            upstream_maj[pid] = maj

    print(f"Loaded {len(upstream_maj)} upstream canonical decisions from {UPSTREAM_FILE.name}")

    files = sorted(list(D2_DIR.glob("stochastic_qwen72b_*.jsonl")))
    total_matches = 0
    total_decisions = 0

    print(f"\nEvaluating {len(files)} stochastic evaluation runs:")
    print(f"{'Filename':<45} | {'Matches':<10} | {'Total':<6} | {'Conformity Rate'}")
    print("-" * 80)

    for fpath in files:
        matches = 0
        total = 0
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pid = rec.get("prompt_id")
                dec = str(rec.get("approval_decision", "")).strip().lower()
                u_maj = upstream_maj.get(pid)
                if u_maj is not None:
                    total += 1
                    if dec == u_maj:
                        matches += 1

        rate = (matches / total) * 100 if total > 0 else 0.0
        print(f"{fpath.name:<45} | {matches:<10} | {total:<6} | {rate:.1f}% ({matches}/{total})")
        total_matches += matches
        total_decisions += total

    overall_rate = (total_matches / total_decisions) * 100 if total_decisions > 0 else 0.0
    print("-" * 80)
    print(f"{'TOTAL / AGGREGATE':<45} | {total_matches:<10} | {total_decisions:<6} | {overall_rate:.1f}% ({total_matches}/{total_decisions})")
    print("=" * 80)

if __name__ == "__main__":
    main()
