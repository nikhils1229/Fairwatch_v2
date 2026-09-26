#!/usr/bin/env python3
"""
FairWatch V2 - Complete 31 Borderline Cells Empirical Analysis Suite
Evaluates the full population of all 31 borderline cells (8,928 multi-agent decisions)
across 24 order permutations and 12 counterfactual applicant identities on Llama-3.1-8B.
Compares the 12-cell post-hoc subset with the remaining 19 cells and the combined 31-cell population.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def analyze_dataset(data_path: Path, prereg_file: Path) -> dict:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    with open(prereg_file, "r", encoding="utf-8") as f:
        orig12_cells = set(int(l.strip()) if l.strip().isdigit() else l.strip() for l in f if l.strip())

    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                records.append({
                    "order": tuple(r["agent_name"] for r in d["agent_records"]),
                    "cell": d["twin_cell_id"],
                    "name": d["applicant_name"],
                    "y": 1.0 if d["approval_decision"] == "approve" else 0.0,
                })

    orders = sorted(list(set(r["order"] for r in records)))
    cells = sorted(list(set(r["cell"] for r in records)))
    names = sorted(list(set(r["name"] for r in records)))

    grid = {(r["cell"], r["order"], r["name"]): r["y"] for r in records}

    # Helper for variance and flip calculations
    def compute_stats(target_cells: list):
        # Flips
        profile_flips = 0
        total_profiles = len(target_cells) * len(names)
        for c in target_cells:
            for n in names:
                decs = [grid[(c, o, n)] for o in orders]
                if len(set(decs)) > 1:
                    profile_flips += 1
        flip_rate = profile_flips / total_profiles if total_profiles > 0 else 0.0

        # Variance
        # Individual-decision variance
        osv_indiv = float(np.mean([np.var([grid[(c, o, i)] for o in orders], ddof=1) for c in target_cells for i in names]))
        csv_indiv = float(np.mean([np.var([grid[(c, o, i)] for i in names], ddof=1) for c in target_cells for o in orders]))
        ratio_indiv = float(osv_indiv / csv_indiv) if csv_indiv > 0 else None

        # Cell-aggregated variance
        osv_cell = float(np.mean([np.var([np.mean([grid[(c, o, i)] for i in names]) for o in orders], ddof=1) for c in target_cells]))
        csv_cell = float(np.mean([np.var([np.mean([grid[(c, o, i)] for o in orders]) for i in names], ddof=1) for c in target_cells]))
        ratio_cell = float(osv_cell / csv_cell) if csv_cell > 0 else None

        # Mean approval rate
        mean_approval = float(np.mean([grid[(c, o, i)] for c in target_cells for o in orders for i in names]))

        return {
            "cell_count": len(target_cells),
            "profile_count": total_profiles,
            "decision_count": len(target_cells) * len(orders) * len(names),
            "flip_count": profile_flips,
            "flip_rate": flip_rate,
            "mean_approval_rate": mean_approval,
            "individual_osv": osv_indiv,
            "individual_csv": csv_indiv,
            "individual_ratio": ratio_indiv,
            "cell_osv": osv_cell,
            "cell_csv": csv_cell,
            "cell_ratio": ratio_cell,
        }

    c_orig12 = [c for c in cells if c in orig12_cells]
    c_rem19 = [c for c in cells if c not in orig12_cells]
    c_all31 = cells

    return {
        "dataset": data_path.name,
        "total_records": len(records),
        "orig_12_cells": compute_stats(c_orig12),
        "rem_19_cells": compute_stats(c_rem19),
        "all_31_cells": compute_stats(c_all31),
    }


def main():
    repo_root = Path(__file__).resolve().parents[2]
    prereg_file = repo_root / "docs/prereg/borderline_cells.txt"
    data_path = repo_root / "results/v2/production/readout_R1/SEQ_full24_llama8b_all31.jsonl"

    summary = analyze_dataset(data_path, prereg_file)
    print(json.dumps(summary, indent=2))

    out_file = repo_root / "analysis/v2/borderline31_analysis_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_file}")


if __name__ == "__main__":
    main()
