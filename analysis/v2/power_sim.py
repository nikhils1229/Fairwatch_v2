"""
FairWatch V2 - Pre-Holdout Power & Precision Simulator (analysis/v2/power_sim.py)
Implements Stage 6.19 precision check:
Simulates whether 144 holdout cells across 6 order permutations have adequate statistical power
to resolve the OSV equivalence margin (delta_osv = 0.01) before unsealing.
(Fable Audit Item 17)
"""

import argparse
import json
import math
import random
import sys
from typing import Dict, Any, Optional

def chi2_quantile_90_lower(df: int) -> float:
    """
    Computes lower 10th percentile of chi-square distribution with df degrees of freedom
    (used to compute upper 90% confidence limit on variance: (df * s^2) / chi2_0.10(df)).
    Uses Wilson-Hilferty transformation.
    """
    if df < 1:
        raise ValueError("Degrees of freedom must be >= 1")
    # For df=5, exact chi2(0.10) = 1.61036
    z = -1.2815515655446004 # 10th percentile of standard normal
    term = 1.0 - 2.0 / (9.0 * df) + z * math.sqrt(2.0 / (9.0 * df))
    return max(0.001, df * (term ** 3))

def run_power_simulation(
    n_cells: int = 144,
    n_sigma: int = 6,
    delta_osv: float = 0.01, # Variance threshold in rate^2 units
    pilot_p: float = 0.50,    # Anchored to pilot approval rate
    sims: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Simulates statistical power of establishing order-schedule variance (OSV) equivalence.
    Under true order invariance, the observed variance across order rates s^2 is purely
    attributable to Bernoulli sampling noise on n_cells.
    Equivalence is declared when upper 90% confidence bound of variance <= delta_osv.
    """
    if n_sigma <= 1:
        raise ValueError("n_sigma must be >= 2 to compute sample variance across order permutations.")

    rng = random.Random(seed)
    df = n_sigma - 1
    chi2_cutoff = chi2_quantile_90_lower(df)

    success_count = 0
    s2_samples = []

    for _ in range(sims):
        # Sample order-specific approval rates for n_cells
        order_rates = []
        for _ in range(n_sigma):
            k = sum(1 for _ in range(n_cells) if rng.random() < pilot_p)
            order_rates.append(k / n_cells)

        mean_r = sum(order_rates) / n_sigma
        s2 = sum((r - mean_r) ** 2 for r in order_rates) / df
        s2_samples.append(s2)

        # Exact upper bound of 90% one-sided CI for true variance
        ci_upper = (df * s2) / chi2_cutoff
        if ci_upper <= delta_osv:
            success_count += 1

    power = success_count / sims
    mean_s2 = sum(s2_samples) / sims if sims > 0 else 0.0

    return {
        "n_cells": n_cells,
        "n_sigma": n_sigma,
        "degrees_of_freedom": df,
        "chi2_0.10_denom": round(chi2_cutoff, 4),
        "delta_osv_variance_threshold": delta_osv,
        "pilot_approval_rate": pilot_p,
        "expected_bernoulli_variance": round((pilot_p * (1.0 - pilot_p)) / n_cells, 6),
        "mean_simulated_s2": round(mean_s2, 6),
        "n_simulations": sims,
        "empirical_power": round(power, 4),
        "gate_passed": power >= 0.80
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=144)
    parser.add_argument("--n-sigma", type=int, default=6)
    parser.add_argument("--delta-osv", type=float, default=0.01)
    parser.add_argument("--pilot-p", type=float, default=0.50)
    parser.add_argument("--sims", type=int, default=2000)
    args = parser.parse_args()

    res = run_power_simulation(
        n_cells=args.n_cells,
        n_sigma=args.n_sigma,
        delta_osv=args.delta_osv,
        pilot_p=args.pilot_p,
        sims=args.sims
    )
    print(json.dumps(res, indent=2))
    if not res["gate_passed"]:
        print(">>> WARNING: Holdout power < 0.80! Expected outcome may be inconclusive.", file=sys.stderr)

if __name__ == "__main__":
    main()
