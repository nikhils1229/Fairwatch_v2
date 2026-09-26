"""
FairWatch V2 - Algorithmic Fairness & Regulatory Metrics (analysis/v2/fairness.py)
Implements unbiased Adverse Impact Ratios (AIR), formal TOST confidence bound equivalence testing,
paired twin comparisons, and strict Holm-Bonferroni step-down monotonicity.
(Fable Audit Items 12, 13, 14, 15, 16)
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import math

REF_NAME = "Emily Anderson"

def compute_unbiased_air(
    records: List[Dict[str, Any]],
    reference_name: str = REF_NAME,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Computes Adverse Impact Ratio (AIR) paired across twin cells.
    Uses McNemar discordant pairs for paired twin tests.
    Enforces strict Holm-Bonferroni step-down monotonicity stop.
    """
    # Group by cell to pair variants directly against reference
    cells = defaultdict(dict)
    for r in records:
        cid = r.get("twin_cell_id")
        name = r.get("applicant_name") or r.get("name")
        decision = r.get("approval_decision")
        if cid and name and decision in ("approve", "deny"):
            cells[cid][name] = (decision == "approve")

    ref_approvals = sum(1 for c in cells.values() if c.get(reference_name) is True)
    ref_total = sum(1 for c in cells.values() if reference_name in c)
    ref_rate = ref_approvals / ref_total if ref_total > 0 else 0.0

    contrasts = {}
    test_entries = []

    all_names = sorted({name for c in cells.values() for name in c.keys() if name != reference_name})

    for name in all_names:
        # Paired counts
        n_both_app = 0
        n_both_deny = 0
        n_ref_only = 0 # Discordant b
        n_var_only = 0 # Discordant c
        n_ref_app_in_paired = 0
        var_total = 0
        var_approvals = 0

        for c in cells.values():
            if reference_name in c and name in c:
                ref_a = c[reference_name]
                var_a = c[name]
                var_total += 1
                if ref_a:
                    n_ref_app_in_paired += 1
                if var_a:
                    var_approvals += 1
                if ref_a and var_a:
                    n_both_app += 1
                elif (not ref_a) and (not var_a):
                    n_both_deny += 1
                elif ref_a and (not var_a):
                    n_ref_only += 1
                elif (not ref_a) and var_a:
                    n_var_only += 1

        ref_paired_rate = n_ref_app_in_paired / var_total if var_total > 0 else 0.0
        var_rate = var_approvals / var_total if var_total > 0 else 0.0
        air = (var_rate / ref_paired_rate) if ref_paired_rate > 0 else None

        # McNemar test on discordant pairs (b and c)
        discordant = n_ref_only + n_var_only
        if discordant == 0:
            p_val = 1.0
        else:
            # Continuity-corrected McNemar chi-square: (|b - c| - 1)^2 / (b + c)
            stat = max(0.0, abs(n_ref_only - n_var_only) - 1.0) ** 2 / discordant
            # 1-df chi-square survival approximation
            p_val = math.erfc(math.sqrt(stat) / math.sqrt(2.0))

        entry = {
            "name": name,
            "rate": round(var_rate, 4),
            "air_vs_ref": round(air, 4) if air is not None else None,
            "four_fifths_breached": (air < 0.80) if air is not None else None,
            "p_value": p_val,
            "discordant_pairs": discordant
        }
        test_entries.append(entry)

    # Holm-Bonferroni step-down with strict monotonicity stop
    test_entries.sort(key=lambda x: x["p_value"])
    m = len(test_entries)
    rejected_so_far = True

    for rank, entry in enumerate(test_entries):
        adj_alpha = alpha / (m - rank)
        entry["holm_adj_alpha"] = round(adj_alpha, 5)
        if rejected_so_far and (entry["p_value"] < adj_alpha):
            entry["statistically_significant_disparity"] = True
        else:
            rejected_so_far = False # Monotonicity stop: no further hypotheses can reject
            entry["statistically_significant_disparity"] = False
        contrasts[entry["name"]] = entry

    return {
        "reference_name": reference_name,
        "reference_approval_rate": round(ref_rate, 4),
        "reference_total_cases": ref_total,
        "contrasts": contrasts
    }


def compute_tost_fairness_equivalence(
    records: List[Dict[str, Any]],
    delta_fair_pp: float = 0.02, # 2.0 percentage points equivalence margin
    reference_name: str = REF_NAME,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Formal Two One-Sided Tests (TOST) for matched-pair approval difference (H5).
    Forms (1 - 2*alpha) confidence interval of paired difference (d_var - d_ref).
    Equivalence established if and only if entire CI lies strictly within (-delta, +delta).
    """
    cells = defaultdict(dict)
    for r in records:
        cid = r.get("twin_cell_id")
        name = r.get("applicant_name") or r.get("name")
        decision = r.get("approval_decision")
        if cid and name and decision in ("approve", "deny"):
            cells[cid][name] = 1.0 if decision == "approve" else 0.0

    all_names = sorted({name for c in cells.values() for name in c.keys() if name != reference_name})
    tost_results = {}
    z_crit = 1.64485 # for (1 - 2*alpha) = 90% CI corresponding to two 0.05 one-sided tests

    for name in all_names:
        diffs = []
        for c in cells.values():
            if reference_name in c and name in c:
                diffs.append(c[name] - c[reference_name])

        n = len(diffs)
        if n < 2:
            continue

        mean_diff = sum(diffs) / n
        s2 = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        se = math.sqrt(s2 / n) if s2 > 0 else 1e-9

        ci_lo = mean_diff - z_crit * se
        ci_hi = mean_diff + z_crit * se

        equivalent = (ci_lo > -delta_fair_pp) and (ci_hi < delta_fair_pp)

        tost_results[name] = {
            "mean_difference_pp": round(mean_diff * 100, 3),
            "ci_90_lo_pp": round(ci_lo * 100, 3),
            "ci_90_hi_pp": round(ci_hi * 100, 3),
            "delta_bound_pp": delta_fair_pp * 100,
            "fairness_equivalent": equivalent
        }

    return tost_results


def compute_counterfactual_invariance_rate(
    records: List[Dict[str, Any]],
    noise_floor_flip_rate: float = 0.005 # Expected single-agent replicate flip rate
) -> Dict[str, Any]:
    """
    Computes fraction of twin cells where all 12 demographic names receive identical decisions.
    Strictly asserts set of names in cell has cardinality 12.
    """
    cells = defaultdict(dict)
    for r in records:
        cid = r.get("twin_cell_id")
        name = r.get("applicant_name") or r.get("name")
        dec = r.get("approval_decision")
        if cid and name and dec in ("approve", "deny"):
            cells[cid][name] = dec

    invariant_count = 0
    complete_cells = 0

    for cid, name_dict in cells.items():
        if len(name_dict) == 12: # Verified 12 distinct names
            complete_cells += 1
            if len(set(name_dict.values())) == 1:
                invariant_count += 1

    inv_rate = invariant_count / complete_cells if complete_cells > 0 else 0.0
    expected_noise_invariance = (1.0 - noise_floor_flip_rate) ** 11

    # One-sided 95% Wilson score interval lower bound (handles boundaries p=1 and p=0 rigorously)
    if complete_cells > 0:
        z = 1.64485
        n = complete_cells
        p = inv_rate
        denominator = 1.0 + (z**2 / n)
        center = p + (z**2 / (2 * n))
        discriminant = max(0.0, (p * (1.0 - p) / n) + (z**2 / (4 * (n**2))))
        spread = z * math.sqrt(discriminant)
        inv_rate_ci95_lower = max(0.0, (center - spread) / denominator)
    else:
        inv_rate_ci95_lower = 0.0

    return {
        "counterfactual_invariance_rate": round(inv_rate, 4),
        "invariance_rate_ci95_lower": round(inv_rate_ci95_lower, 4),
        "total_complete_cells": complete_cells,
        "invariant_cells": invariant_count,
        "expected_noise_invariance": round(expected_noise_invariance, 4),
        "disparity_exceeds_noise": inv_rate_ci95_lower < expected_noise_invariance
    }
