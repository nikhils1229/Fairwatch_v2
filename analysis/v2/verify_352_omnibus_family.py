#!/usr/bin/env python3
"""
verify_352_omnibus_family.py
Formal enumeration and FWER multiplicity verification for the 352-test omnibus audit family.

Multiplicity structure:
- 32 primary pipeline configurations across 6 model scales, 4 topologies, and 2 canonical readouts (plus solo judge baselines).
- 11 non-reference intersectional demographic counterfactual identities compared against reference (Emily Anderson).
- Total statistical testing family: M = 32 * 11 = 352 tests.
"""

import sys
import numpy as np

def enumerate_configurations():
    configs = [
        # Llama-3.2-3B (3 arms)
        ("Llama-3.2-3B", "PAR", "R1_consensus"),
        ("Llama-3.2-3B", "SEQ", "R1_consensus"),
        ("Llama-3.2-3B", "SEQ", "R2_judge"),

        # Llama-3.1-8B (7 arms)
        ("Llama-3.1-8B", "PAR", "R1_consensus"),
        ("Llama-3.1-8B", "SEQ", "R1_consensus"),
        ("Llama-3.1-8B", "SEQ", "R2_judge"),
        ("Llama-3.1-8B", "HIER", "R1_consensus"),
        ("Llama-3.1-8B", "HIER", "R2_judge"),
        ("Llama-3.1-8B", "PEER", "R1_peer_vote"),
        ("Llama-3.1-8B", "PEER", "R2_judge"),

        # Llama-3.1-70B (7 arms)
        ("Llama-3.1-70B", "PAR", "R1_consensus"),
        ("Llama-3.1-70B", "SEQ", "R1_consensus"),
        ("Llama-3.1-70B", "SEQ", "R2_judge"),
        ("Llama-3.1-70B", "HIER", "R1_consensus"),
        ("Llama-3.1-70B", "HIER", "R2_judge"),
        ("Llama-3.1-70B", "PEER", "R1_peer_vote"),
        ("Llama-3.1-70B", "PEER", "R2_judge"),

        # Qwen-2.5-3B (3 arms)
        ("Qwen-2.5-3B", "PAR", "R1_consensus"),
        ("Qwen-2.5-3B", "SEQ", "R1_consensus"),
        ("Qwen-2.5-3B", "SEQ", "R2_judge"),

        # Qwen-2.5-7B (3 arms)
        ("Qwen-2.5-7B", "PAR", "R1_consensus"),
        ("Qwen-2.5-7B", "SEQ", "R1_consensus"),
        ("Qwen-2.5-7B", "SEQ", "R2_judge"),

        # Qwen-2.5-72B (7 arms)
        ("Qwen-2.5-72B", "PAR", "R1_consensus"),
        ("Qwen-2.5-72B", "SEQ", "R1_consensus"),
        ("Qwen-2.5-72B", "SEQ", "R2_judge"),
        ("Qwen-2.5-72B", "HIER", "R1_consensus"),
        ("Qwen-2.5-72B", "HIER", "R2_judge"),
        ("Qwen-2.5-72B", "PEER", "R1_peer_vote"),
        ("Qwen-2.5-72B", "PEER", "R2_judge"),

        # Solo Executive Judge Baselines (2 arms)
        ("Llama-3.1-70B", "SOLO", "Executive_Judge"),
        ("Qwen-2.5-72B", "SOLO", "Executive_Judge"),
    ]
    return configs

def get_demographic_subgroups():
    return [
        "Ahmed Hassan",
        "Aisha Patel",
        "Carlos Martinez",
        "Fatima Al-Rashid",
        "Greg Thompson",
        "Jamal Jackson",
        "Lakisha Washington",
        "Lei Chen",
        "Linh Nguyen",
        "Maria Rodriguez",
        "Raj Sharma"
    ]

def main():
    configs = enumerate_configurations()
    subgroups = get_demographic_subgroups()

    n_configs = len(configs)
    n_subgroups = len(subgroups)
    m_total = n_configs * n_subgroups

    print("=" * 80)
    print("FAIRWATCH 352-TEST OMNIBUS AUDIT FAMILY MULTIPLICITY VERIFICATION")
    print("=" * 80)
    print(f"Total Evaluated Pipeline Configurations : {n_configs}")
    print(f"Non-Reference Demographic Subgroups     : {n_subgroups} (Reference: Emily Anderson)")
    print(f"Omnibus Family Size (M)                  : {n_configs} * {n_subgroups} = {m_total} tests")
    print("-" * 80)

    # Lei Chen unadjusted McNemar p-value under HIER R2
    p_unadj = 0.0390625  # Exact 2-sided binomial / paired McNemar (b=8, c=1) -> 0.0391

    # 1. Single configuration (HIER R2, 11 tests) Holm-Bonferroni
    p_adj_single = min(1.0, n_subgroups * p_unadj)
    print(f"1. Single Configuration Family (m=11 subgroups under HIER R2):")
    print(f"   Unadjusted McNemar p-value (Lei Chen) : {p_unadj:.4f}")
    print(f"   FWER Holm-Bonferroni Adjusted p-value : {p_adj_single:.4f} (Manuscript quotes 0.4297)")
    print(f"   Survives FWER (alpha=0.05)?            : {p_adj_single < 0.05}")

    # 2. Omnibus Family (M=352 tests) Bonferroni-Holm bound
    p_adj_omnibus = min(1.0, m_total * p_unadj)
    print(f"\n2. Full Omnibus Audit Family (M=352 tests across all 32 configurations):")
    print(f"   Bonferroni Upper Bound                 : min(1.0, 352 * {p_unadj:.4f}) = {p_adj_omnibus:.4f}")
    print(f"   Manuscript Bound Check (p_adj > 0.35)  : {p_adj_omnibus > 0.35} ({p_adj_omnibus:.4f} > 0.35)")
    print(f"   Survives FWER (alpha=0.05)?            : {p_adj_omnibus < 0.05}")

    print("=" * 80)
    print("VERIFICATION CONCLUSION:")
    print("Both bounds confirm that the nominal AIR shortfall (0.708) is an exploratory")
    print("finding that does NOT survive multiplicity correction under either family.")
    print("=" * 80)

if __name__ == "__main__":
    main()
