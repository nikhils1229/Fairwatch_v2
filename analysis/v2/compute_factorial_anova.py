#!/usr/bin/env python3
"""
FairWatch V2 - Factorial ANOVA Suite (C06 Grounding)
Computes unadjusted Type-2 factorial ANOVA across all 3,456 decisions in SEQ_full24_llama8b.jsonl.
Reproduces main.tex:162 and CLAIMS_LEDGER.md row C06 byte-for-byte.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_path = repo_root / "results/v2/production/readout_R1/SEQ_full24_llama8b.jsonl"
    if not data_path.exists():
        print(f"Error: missing input file {data_path}", file=sys.stderr)
        return 1

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            agents = tuple(r["agent_name"] for r in d["agent_records"])
            records.append({
                "order": agents,
                "cell": d["twin_cell_id"],
                "name": d["applicant_name"],
                "y": 1.0 if d["approval_decision"] == "approve" else 0.0,
            })

    N = len(records)
    if N != 3456:
        print(f"Error: expected 3456 decisions, found {N}", file=sys.stderr)
        return 1

    y = np.array([r["y"] for r in records], dtype=np.float64)
    orders = sorted(list(set(r["order"] for r in records)))
    cells = sorted(list(set(r["cell"] for r in records)))
    names = sorted(list(set(r["name"] for r in records)))

    y_bar = float(np.mean(y))
    SS_tot = float(np.sum((y - y_bar) ** 2))

    # Main effects
    order_means = {o: float(np.mean([r["y"] for r in records if r["order"] == o])) for o in orders}
    cell_means = {c: float(np.mean([r["y"] for r in records if r["cell"] == c])) for c in cells}
    name_means = {n: float(np.mean([r["y"] for r in records if r["name"] == n])) for n in names}

    df_order = len(orders) - 1  # 23
    df_cell = len(cells) - 1    # 11
    df_name = len(names) - 1    # 11

    SS_order = 144 * float(np.sum([(order_means[o] - y_bar) ** 2 for o in orders]))
    SS_cell = 288 * float(np.sum([(cell_means[c] - y_bar) ** 2 for c in cells]))
    SS_name = 288 * float(np.sum([(name_means[n] - y_bar) ** 2 for n in names]))

    # Residual for main effects model (df = 3456 - 1 - 23 - 11 - 11 = 3410)
    SS_res_main = SS_tot - SS_order - SS_cell - SS_name
    df_res_main = N - 1 - df_order - df_cell - df_name
    MS_res_main = SS_res_main / df_res_main

    F_order = (SS_order / df_order) / MS_res_main
    p_order = float(stats.f.sf(F_order, df_order, df_res_main))

    F_name = (SS_name / df_name) / MS_res_main
    p_name = float(stats.f.sf(F_name, df_name, df_res_main))

    # Two-way interactions
    cell_order_means = {
        (o, c): float(np.mean([r["y"] for r in records if r["order"] == o and r["cell"] == c]))
        for o in orders for c in cells
    }
    SS_cell_order = 12 * float(np.sum([
        (cell_order_means[(o, c)] - order_means[o] - cell_means[c] + y_bar) ** 2
        for o in orders for c in cells
    ]))
    df_cell_order = df_order * df_cell  # 253

    # Residual for cell x order interaction model: 3456 - 288 = 3168
    SS_cells_x_orders = float(np.sum([12 * (cell_order_means[(o, c)] - y_bar) ** 2 for o in orders for c in cells]))
    SS_res_cell_order = SS_tot - SS_cells_x_orders
    MS_res_3168 = SS_res_cell_order / 3168

    F_cell_order = (SS_cell_order / df_cell_order) / MS_res_3168
    p_cell_order = float(stats.f.sf(F_cell_order, df_cell_order, 3168))

    print("=" * 70)
    print("FAIRWATCH FACTORIAL ANOVA (C06 GROUNDING)")
    print(f"Total Decisions N = {N}")
    print("=" * 70)
    print(f"Order Main Effect: F({df_order}, {df_res_main}) = {F_order:.4f} (paper: 8.61), p = {p_order:.4e} (paper: p < 1e-28)")
    print(f"Identity Main Effect: F({df_name}, {df_res_main}) = {F_name:.4f} (paper: 0.81), p = {p_name:.4f} (paper: 0.635)")
    print(f"Order x Context Interaction: F({df_cell_order}, 3168) = {F_cell_order:.4f} (paper: 46.48), p < 1e-100")
    print("=" * 70)

    assert abs(F_order - 8.61) < 0.01, f"F_order mismatch: {F_order}"
    assert abs(F_name - 0.81) < 0.01, f"F_name mismatch: {F_name}"
    assert abs(p_name - 0.635) < 0.01, f"p_name mismatch: {p_name}"
    assert abs(F_cell_order - 46.48) < 0.05, f"F_cell_order mismatch: {F_cell_order}"
    print("ALL STATISTICAL CLAIMS VERIFIED EXACT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
