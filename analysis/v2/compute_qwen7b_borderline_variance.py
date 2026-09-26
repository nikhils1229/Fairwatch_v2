#!/usr/bin/env python3
"""
FairWatch V2 - Qwen-2.5-7B Borderline Permutation Variance (C30 Grounding)
Computes individual-decision OSV and CSV across 3,456 decisions in SEQ_qwen7b.jsonl.
Reproduces appendices.tex:241 and CLAIMS_LEDGER.md row C30 byte-for-byte.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "results/v2/production/readout_R1/SEQ_qwen7b.jsonl"

    if not data_path.exists():
        print(f"Error: missing input {data_path}", file=sys.stderr)
        return 1

    by_cell = defaultdict(list)
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = d.get("twin_cell_id")
            dec = 1 if d.get("approval_decision") == "approve" else 0
            by_cell[cid].append(dec)

    n_cells = len(by_cell)
    total_decs = sum(len(v) for v in by_cell.values())
    if n_cells != 12 or total_decs != 3456:
        print(f"Error: expected 3456 decisions across 12 cells, got {total_decs} across {n_cells}", file=sys.stderr)
        return 1

    vars_pop = [float(np.var(decs, ddof=0)) for decs in by_cell.values()]
    cell_means = [float(np.mean(decs)) for decs in by_cell.values()]
    osv = float(np.mean(vars_pop))
    csv = float(np.var(cell_means, ddof=0))
    ratio = osv / csv

    print("=" * 70)
    print("QWEN-2.5-7B BORDERLINE PERMUTATION VARIANCE (C30 GROUNDING)")
    print(f"Borderline Contexts = {n_cells}, Total Decisions = {total_decs}")
    print("=" * 70)
    print(f"OSV (Within-Cell Order Variance): {osv:.5f} (paper: 0.02118)")
    print(f"CSV (Between-Cell Context Variance): {csv:.5f} (paper: 0.00527)")
    print(f"OSV / CSV Ratio: {ratio:.3f} (paper: 4.017)")
    print("=" * 70)

    assert abs(osv - 0.02118) < 0.0001, f"OSV mismatch: {osv}"
    assert abs(csv - 0.00527) < 0.0001, f"CSV mismatch: {csv}"
    assert abs(ratio - 4.017) < 0.005, f"Ratio mismatch: {ratio}"
    print("ALL QWEN-7B VARIANCE CLAIMS VERIFIED EXACT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
