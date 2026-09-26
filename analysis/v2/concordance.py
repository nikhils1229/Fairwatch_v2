"""
FairWatch V2 - Replication Concordance Table Evaluator (analysis/v2/concordance.py)
Evaluates sign concordance and confidence interval overlap between exploratory and confirmatory holdout estimates.
(Fable Audit Item 19)
"""

from typing import Dict, Any, List, Optional

def evaluate_replication_concordance(
    exploratory_metrics: Dict[str, Any],
    confirmatory_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluates replication success for each preregistered estimand:
    - Sign agreement (direction, strictly handling zero-boundaries)
    - 95% CI overlap (valid interval comparison)
    - Effect size retention ratio
    """
    concordance_table = {}
    total_metrics = 0
    sign_replications = 0
    ci_overlaps = 0
    ci_evaluated = 0

    common_keys = set(exploratory_metrics.keys()) & set(confirmatory_metrics.keys())
    for key in sorted(common_keys):
        exp = exploratory_metrics[key]
        conf = confirmatory_metrics[key]

        if not (isinstance(exp, dict) and isinstance(conf, dict)):
            continue

        e_val = exp.get("estimate")
        c_val = conf.get("estimate")
        e_ci = exp.get("ci_95")
        c_ci = conf.get("ci_95")

        if e_val is None or c_val is None:
            continue

        total_metrics += 1

        # Precise zero handling for sign match
        if e_val == 0.0 and c_val == 0.0:
            sign_match = True
        elif e_val == 0.0 or c_val == 0.0:
            sign_match = False
        else:
            sign_match = (e_val > 0 and c_val > 0) or (e_val < 0 and c_val < 0)

        # CI overlap requires valid 2-element tuples
        if (
            isinstance(e_ci, (list, tuple)) and len(e_ci) == 2 and
            isinstance(c_ci, (list, tuple)) and len(c_ci) == 2
        ):
            ci_overlap = max(e_ci[0], c_ci[0]) <= min(e_ci[1], c_ci[1])
            ci_evaluated += 1
            if ci_overlap:
                ci_overlaps += 1
        else:
            ci_overlap = None

        if sign_match:
            sign_replications += 1

        concordance_table[key] = {
            "exploratory_estimate": e_val,
            "confirmatory_estimate": c_val,
            "sign_concordant": sign_match,
            "ci_overlapping": ci_overlap,
            "retention_ratio": round(c_val / e_val, 4) if e_val != 0 else None
        }

    return {
        "total_estimands_evaluated": total_metrics,
        "sign_replication_rate": round(sign_replications / total_metrics, 4) if total_metrics > 0 else None,
        "ci_overlap_rate": round(ci_overlaps / ci_evaluated, 4) if ci_evaluated > 0 else None,
        "concordance_details": concordance_table
    }
