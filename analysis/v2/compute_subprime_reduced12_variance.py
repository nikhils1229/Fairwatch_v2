#!/usr/bin/env python3
"""
FairWatch V2 - Subprime Reduced-12 Permutation Variance (C29 Grounding)
Computes individual-decision OSV and CSV across 2,880 Subprime decisions.
Reproduces main.tex:176 and CLAIMS_LEDGER.md row C29 byte-for-byte.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    meta_path = repo_root / "data/derived/core_benchmark_v2.csv"
    data_path = repo_root / "results/v2/production/readout_R1/SEQ_reduced12_llama8b.jsonl"

    if not meta_path.exists() or not data_path.exists():
        print("Error: missing inputs", file=sys.stderr)
        return 1

    df_meta = pd.read_csv(meta_path)
    subprime_cells = set(df_meta[df_meta["credit_band"] == "subprime_nearprime"]["twin_cell_id"].unique())

    by_cell = defaultdict(list)
    all_decs = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d.get("twin_cell_id")
            if cid in subprime_cells:
                dec = 1 if d.get("approval_decision") == "approve" else 0
                by_cell[cid].append(dec)
                all_decs.append(dec)

    n_decisions = len(all_decs)
    n_cells = len(by_cell)
    if n_decisions != 2880 or n_cells != 20:
        print(f"Error: expected 2880 decisions across 20 cells, got {n_decisions} across {n_cells}", file=sys.stderr)
        return 1

    vars_pop = [float(np.var(decs, ddof=0)) for decs in by_cell.values()]
    cell_means = [float(np.mean(decs)) for decs in by_cell.values()]
    osv = float(np.mean(vars_pop))
    csv = float(np.var(cell_means, ddof=0))
    ratio = osv / csv
    mean_app = float(np.mean(all_decs))

    print("=" * 70)
    print("SUBPRIME REDUCED-12 PERMUTATION VARIANCE (C29 GROUNDING)")
    print(f"Subprime Contexts = {n_cells}, Total Decisions = {n_decisions}")
    print("=" * 70)
    print(f"Mean Approval Rate: {mean_app * 100:.2f}% ({sum(all_decs)}/{n_decisions}) (paper: 50.73%, 1461/2880)")
    print(f"OSV (Within-Cell Order Variance): {osv:.5f} (paper: 0.12302)")
    print(f"CSV (Between-Cell Context Variance): {csv:.5f} (paper: 0.12693)")
    print(f"OSV / CSV Ratio: {ratio:.3f} (paper: 0.969)")
    print("=" * 70)

    assert abs(osv - 0.12302) < 0.0001, f"OSV mismatch: {osv}"
    assert abs(csv - 0.12693) < 0.0001, f"CSV mismatch: {csv}"
    assert abs(ratio - 0.969) < 0.001, f"Ratio mismatch: {ratio}"
    assert sum(all_decs) == 1461, f"Sum mismatch: {sum(all_decs)}"
    print("ALL SUBPRIME VARIANCE CLAIMS VERIFIED EXACT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
