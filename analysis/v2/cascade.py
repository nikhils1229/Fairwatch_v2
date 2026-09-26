"""
FairWatch V2 - De-confounded Informational Cascade Estimator (analysis/v2/cascade.py)
Implements position-conditional private-signal abandonment rate:
abandonment[position][n_upstream_agree]
Stratified by solo_margin terciles to separate true informational cascades from hard-case ambiguity.
(Fable Audit Item 1)
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

def compute_position_conditional_cascades(
    sequential_records: List[Dict[str, Any]],
    solo_verdicts: Dict[Tuple[str, str], str], # (case_id, persona) -> solo_verdict
    solo_margins: Optional[Dict[str, float]] = None # case_id -> solo_margin logodds
) -> Dict[str, Any]:
    """
    Computes abandonment rate of private (solo) verdict conditional on upstream consensus,
    strictly partitioned by position k in {2, 3, 4} and stratified by solo margin.
    """
    # abandonment[stratum][pos][n_agree] -> {"total": int, "abandoned": int}
    matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"total": 0, "abandoned": 0})))
    missing_pos_count = 0
    unparseable_count = 0
    missing_solo_count = 0
    missing_upstream_count = 0

    # Empirical terciles from solo margins if provided
    cut_low = 0.5
    cut_high = 1.5
    if solo_margins:
        vals = sorted(abs(v) for v in solo_margins.values())
        if len(vals) >= 3:
            cut_low = vals[len(vals) // 3]
            cut_high = vals[(2 * len(vals)) // 3]

    for rec in sequential_records:
        pos = rec.get("seq_position")
        if pos is None:
            missing_pos_count += 1
            continue
        if pos == 1:
            continue # Position 1 has no upstream agents

        case_id = rec.get("twin_cell_id") or rec.get("case_id")
        agent_name = rec.get("agent_name")
        actual_verdict = rec.get("approval_decision")

        # Check upstream verdicts presence
        if rec.get("upstream_verdicts") is None:
            missing_upstream_count += 1
            continue
        upstream_verdicts = rec["upstream_verdicts"]

        # Track unparseable records
        if actual_verdict not in ("approve", "deny"):
            unparseable_count += 1
            continue

        # Lookup private baseline
        solo_v = solo_verdicts.get((case_id, agent_name))
        if solo_v is None or solo_v not in ("approve", "deny"):
            missing_solo_count += 1
            continue

        # Determine margin stratum
        margin_stratum = "unstratified"
        if solo_margins and case_id in solo_margins:
            m = abs(solo_margins[case_id])
            margin_stratum = "hard" if m < cut_low else "moderate" if m < cut_high else "clear"

        n_agree = sum(1 for uv in upstream_verdicts if uv == solo_v)

        matrix[margin_stratum][pos][n_agree]["total"] += 1
        if actual_verdict != solo_v:
            matrix[margin_stratum][pos][n_agree]["abandoned"] += 1

    rates = {}
    for stratum, pos_map in sorted(matrix.items()):
        rates[stratum] = {}
        for pos, agree_map in sorted(pos_map.items()):
            rates[stratum][f"position_{pos}"] = {}
            for n_agree, counts in sorted(agree_map.items()):
                tot = counts["total"]
                ab = counts["abandoned"]
                r = ab / tot if tot > 0 else 0.0
                rates[stratum][f"position_{pos}"][f"agree_{n_agree}"] = {
                    "abandonment_rate": r,
                    "n_abandoned": ab,
                    "n_total": tot
                }

    rates["_diagnostics"] = {
        "missing_seq_position_records": missing_pos_count,
        "missing_upstream_verdicts_records": missing_upstream_count,
        "unparseable_records": unparseable_count,
        "missing_solo_baseline_records": missing_solo_count
    }

    return rates
