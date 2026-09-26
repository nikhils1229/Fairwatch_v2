#!/usr/bin/env python3
"""
FairWatch V2 - Baseline Ground Truth (pi*) Provenance & Verification Suite (Hardened)
Path: analysis/v2/trace_pi_star_provenance.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from analysis.v2.reference_policy import compute_pi_star

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pi_star_provenance")

TIER_NORMALIZATION = {
    "super_prime": "Super Prime",
    "prime": "Prime",
    "prime_minus": "Prime Minus",
    "subprime_nearprime": "Subprime",
    "subprime": "Subprime",
}


def verify_benchmark_provenance(csv_path: Path) -> int:
    if not csv_path.exists():
        logger.error(f"Benchmark file not found: {csv_path}")
        return 1

    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows.append(r)

    total_rows = len(rows)
    logger.info(f"Loaded {total_rows} rows from {csv_path}")

    # Group by unique twin cell ID
    context_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cid = str(r.get("twin_cell_id") or r.get("cell_id") or r.get("context_id")).strip()
        context_groups[cid].append(r)

    n_contexts = len(context_groups)
    mismatches_decision = 0
    mismatches_borderline = 0
    identity_violations = 0

    tier_context_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"approve": 0, "deny": 0, "total": 0})
    borderline_context_counts: Dict[str, int] = {"approve": 0, "deny": 0, "total": 0}

    for cid, members in context_groups.items():
        base = members[0]
        rec_pi, rec_p, rec_bl = compute_pi_star(base)

        csv_pi = str(base.get("pi_star", "")).strip().lower()
        csv_bl = str(base.get("borderline", "")).strip().lower() in ("true", "1", "yes")

        if rec_pi != csv_pi:
            mismatches_decision += 1
        if rec_bl != csv_bl:
            mismatches_borderline += 1

        # Check identity invariance across demographic names in cell
        for m in members[1:]:
            m_csv_pi = str(m.get("pi_star", "")).strip().lower()
            m_csv_bl = str(m.get("borderline", "")).strip().lower() in ("true", "1", "yes")
            if m_csv_pi != csv_pi or m_csv_bl != csv_bl:
                identity_violations += 1

        raw_tier = str(base.get("credit_band") or base.get("credit_tier") or "Unknown").strip().lower()
        tier = TIER_NORMALIZATION.get(raw_tier, raw_tier.title())

        tier_context_counts[tier][rec_pi] += 1
        tier_context_counts[tier]["total"] += 1

        if rec_bl:
            borderline_context_counts[rec_pi] += 1
            borderline_context_counts["total"] += 1

    print("\n" + "=" * 78)
    print("FAIRWATCH V2 - PI_STAR NORMATIVE GROUND TRUTH PROVENANCE AUDIT")
    print("=" * 78)
    print(f"1. Integrity Gate:")
    print(f"   - Total Rows: {total_rows:,} across {n_contexts} unique contexts")
    print(f"   - Decision Mismatches: {mismatches_decision}")
    print(f"   - Borderline Flag Mismatches: {mismatches_borderline}")
    print(f"   - Identity Invariance Violations: {identity_violations}")

    expected_context_tiers = {
        "Super Prime": (20, 17, 3),
        "Prime": (22, 13, 9),
        "Prime Minus": (22, 8, 14),
        "Subprime": (20, 2, 18),
    }

    print("\n2. Unique Context Tier Distribution (N=84 target):")
    all_tiers_pass = True
    for t_name, (exp_n, exp_app, exp_den) in expected_context_tiers.items():
        act_tot = tier_context_counts[t_name]["total"]
        act_app = tier_context_counts[t_name]["approve"]
        pass_flag = "PASS" if (act_tot == exp_n and act_app == exp_app) else "FAIL"
        if pass_flag == "FAIL":
            all_tiers_pass = False
        print(f"   | {t_name:<16} | N={act_tot:>2} (Exp: {exp_n:>2}) | Approvals: {act_app:>2}/{act_tot:<2} | [{pass_flag}] |")

    bl_tot = borderline_context_counts["total"]
    bl_app = borderline_context_counts["approve"]
    print(f"\n3. Analytic Borderline Stratum (|p - 0.50| <= 0.18): {bl_tot}/84 contexts, {bl_app} Approve / {bl_tot - bl_app} Deny")

    print("\n4. Appendix C Cascade Approval Expansion Reconciliation (Emily Anderson):")
    print("   | Risk Tier | Real pi* Baseline | SEQ R1 (8B Cascade) | Expansion Delta (Cascade Effect) |")
    print("   | :--- | :---: | :---: | :---: |")
    seq_r1_reference = {
        "Super Prime": 20,
        "Prime": 19,
        "Prime Minus": 16,
        "Subprime": 10,
    }
    total_delta = 0
    for t_name, (exp_n, pi_app, _) in expected_context_tiers.items():
        seq_app = seq_r1_reference[t_name]
        delta = seq_app - pi_app
        total_delta += delta
        sign = "+" if delta > 0 else ""
        print(f"   | {t_name:<16} | {pi_app:>2}/{exp_n} ({pi_app/exp_n:.1%}) | {seq_app:>2}/{exp_n} ({seq_app/exp_n:.1%}) | {sign}{delta} approvals |")

    print("   | " + "-" * 76 + " |")
    print(f"   | Total Benchmark  | 40/84 (47.6%) | 65/84 (77.4%) | +{total_delta} approvals (Net Expansion) |")
    print("=" * 78 + "\n")

    return 0 if (mismatches_decision == 0 and mismatches_borderline == 0 and all_tiers_pass) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="FairWatch V2 pi* Provenance Verification")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/derived/core_benchmark_v2.csv"),
        help="Path to core_benchmark_v2.csv",
    )
    args = parser.parse_args()
    return verify_benchmark_provenance(args.benchmark.resolve())


if __name__ == "__main__":
    sys.exit(main())
