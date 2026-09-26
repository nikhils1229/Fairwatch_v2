#!/usr/bin/env python3
"""
FairWatch V2 - Reference Policy Scorecard (analysis/v2/reference_policy.py)
Synthetic reference policies π* and π*₂ for credit decision evaluation.
Strictly excludes degenerate dti_ratio; uses lti_ratio.
"""

import math
from typing import Dict, Any, Tuple

VISA_WEIGHTS_PI1 = {
    "US Citizen": 0.35,
    "Permanent Resident or Green Card": 0.20,
    "H-1B Visa Holder": 0.0,
    "F-1 Student Visa": -0.40,
}

VISA_WEIGHTS_PI2 = {
    "US Citizen": 0.40,
    "Permanent Resident or Green Card": 0.15,
    "H-1B Visa Holder": -0.10,
    "F-1 Student Visa": -0.60,
}

def compute_pi_star(row: Dict[str, Any]) -> Tuple[str, float, bool]:
    """
    Computes primary reference policy π*(x).
    Returns (decision, probability, is_borderline).
    """
    cs = float(row["credit_score"])
    inc = float(row["income"])
    la = float(row["loan_amount"])
    age = float(row["age"])
    visa = str(row["visa_status"]).strip()
    lti = float(la / inc) if inc > 0 else 1.0

    z = (
        0.15
        + 1.25 * ((cs - 680.0) / 50.0)
        + 0.50 * ((inc - 65000.0) / 30000.0)
        - 1.10 * (lti - 1.0)
        + VISA_WEIGHTS_PI1.get(visa, 0.0)
        + 0.10 * ((age - 40.0) / 15.0)
    )
    p = 1.0 / (1.0 + math.exp(-z))
    decision = "approve" if p >= 0.50 else "deny"
    is_borderline = abs(p - 0.50) <= 0.18
    return decision, p, is_borderline

def compute_pi_star_2(row: Dict[str, Any]) -> Tuple[str, float, bool]:
    """
    Computes secondary alternative reference policy π*₂(x) for sensitivity analysis.
    """
    cs = float(row["credit_score"])
    inc = float(row["income"])
    la = float(row["loan_amount"])
    age = float(row["age"])
    visa = str(row["visa_status"]).strip()
    lti = float(la / inc) if inc > 0 else 1.0

    z = (
        0.05
        + 1.35 * ((cs - 680.0) / 50.0)
        + 0.40 * ((inc - 65000.0) / 30000.0)
        - 1.30 * (lti - 1.0)
        + VISA_WEIGHTS_PI2.get(visa, 0.0)
        + 0.05 * ((age - 40.0) / 15.0)
    )
    p = 1.0 / (1.0 + math.exp(-z))
    decision = "approve" if p >= 0.55 else "deny"
    is_borderline = abs(p - 0.55) <= 0.18
    return decision, p, is_borderline
