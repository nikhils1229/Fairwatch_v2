#!/usr/bin/env python3
"""
FairWatch V2 - Inter-Model Decision Concordance (Cohen's Kappa) Analysis (Hardened)
Path: analysis/v2/compute_model_concordance_kappa.py

Evaluates pairwise Cohen's kappa for model pairs across credit risk strata:
1. Clear-Approval (borderline == False and pi_star == 'approve')
2. Clear-Rejection (borderline == False and pi_star == 'deny')
3. Borderline (borderline == True)
4. Overall (All 1,008 cases)

Model Pairs:
- Llama-3.2-3B vs Llama-3.1-8B
- Llama-3.1-8B vs Llama-3.1-70B
- Qwen2.5-7B vs Qwen2.5-72B
- Llama-3.1-70B vs Qwen2.5-72B
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("concordance_kappa")


@dataclass(frozen=True)
class BenchmarkCell:
    twin_cell_id: str
    applicant_name: str
    credit_tier: str
    pi_star: str
    borderline: bool


@dataclass
class KappaResult:
    n_samples: int
    po: float
    pe: float
    kappa: float
    is_undefined: bool
    status_label: str
    n_both_approve: int
    n_both_deny: int
    n_discordant_10: int
    n_discordant_01: int


def load_benchmark(csv_path: Path) -> Dict[str, BenchmarkCell]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found at {csv_path}")

    cells: Dict[str, BenchmarkCell] = {}
    with open(csv_path, "r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            tcid = row.get("twin_cell_id", "").strip()
            name = (row.get("name") or row.get("applicant_name") or "").strip()
            if not tcid or not name:
                continue

            bl_raw = str(row.get("borderline", "false")).strip().lower()
            cell = BenchmarkCell(
                twin_cell_id=tcid,
                applicant_name=name,
                credit_tier=row.get("credit_band") or row.get("credit_tier", "Unknown"),
                pi_star=row.get("pi_star", "deny").strip().lower(),
                borderline=bl_raw in ("true", "1", "yes", "t"),
            )
            cells[f"{tcid}::{name}"] = cell
            pid = str(row.get("prompt_id") or "").strip()
            if pid:
                cells[pid] = cell
    return cells


def load_decisions(jsonl_path: Path) -> Dict[str, int]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Decision JSONL not found at {jsonl_path}")

    decisions: Dict[str, int] = {}
    dropped_parse_errors = 0

    with open(jsonl_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                dropped_parse_errors += 1
                continue

            tcid = str(data.get("twin_cell_id") or data.get("cell_id") or "").strip()
            name = str(data.get("applicant_name") or data.get("name") or "").strip()
            pid = str(data.get("prompt_id") or data.get("id") or "").strip()

            if not tcid or not name:
                if "_" in pid:
                    parts = pid.split("_", 1)
                    tcid = tcid or parts[0]
                    name = name or parts[1]

            raw_dec = data.get("approval_decision")
            if raw_dec is None:
                raw_dec = data.get("decision") or data.get("consensus") or data.get("executive_decision")
            if raw_dec is None and "approved" in data:
                raw_dec = "approve" if data["approved"] else "deny"

            if raw_dec is None:
                dropped_parse_errors += 1
                continue

            clean_dec = str(raw_dec).strip().lower()
            if clean_dec in ("approve", "approved", "1", "true"):
                val = 1
            elif clean_dec in ("deny", "denied", "reject", "rejected", "0", "false"):
                val = 0
            else:
                dropped_parse_errors += 1
                continue

            if tcid and name:
                decisions[f"{tcid}::{name}"] = val
            if pid:
                decisions[pid] = val

    if dropped_parse_errors > 0:
        logger.warning(f"{jsonl_path.name}: Excluded {dropped_parse_errors} unparseable/missing records.")
    return decisions


def compute_concordance(y1: List[int], y2: List[int]) -> KappaResult:
    n = len(y1)
    if n == 0 or len(y2) != n:
        return KappaResult(0, 0.0, 0.0, float("nan"), True, "N/A", 0, 0, 0, 0)

    a = sum(1 for a_i, b_i in zip(y1, y2) if a_i == 1 and b_i == 1)
    b = sum(1 for a_i, b_i in zip(y1, y2) if a_i == 1 and b_i == 0)
    c = sum(1 for a_i, b_i in zip(y1, y2) if a_i == 0 and b_i == 1)
    d = sum(1 for a_i, b_i in zip(y1, y2) if a_i == 0 and b_i == 0)

    po = (a + d) / n
    p1 = (a + b) / n
    q1 = (c + d) / n
    p2 = (a + c) / n
    q2 = (b + d) / n
    pe = (p1 * p2) + (q1 * q2)

    if (a == n and d == 0) or (d == n and a == 0) or (1.0 - pe) <= 1e-12:
        return KappaResult(n, po, pe, float("nan"), True, "NaN (Unanimous)", a, d, b, c)

    kappa = (po - pe) / (1.0 - pe)
    return KappaResult(n, po, pe, kappa, False, f"{kappa:.3f}", a, d, b, c)


class ConcordanceEvaluator:
    MODEL_MAP = {
        "llama3b": "Llama-3.2-3B",
        "llama8b": "Llama-3.1-8B",
        "llama70b": "Llama-3.1-70B",
        "qwen7b": "Qwen2.5-7B",
        "qwen72b": "Qwen2.5-72B",
    }

    PAIRS = [
        ("llama3b", "llama8b", "Llama-3.2-3B vs. Llama-3.1-8B"),
        ("llama8b", "llama70b", "Llama-3.1-8B vs. Llama-3.1-70B"),
        ("qwen7b", "qwen72b", "Qwen2.5-7B vs. Qwen2.5-72B"),
        ("llama70b", "qwen72b", "Llama-3.1-70B vs. Qwen2.5-72B"),
    ]

    def __init__(self, benchmark_path: Path, readout_dir: Path, allow_r3_fallback: bool = False) -> None:
        self.benchmark = load_benchmark(benchmark_path)
        self.readout_dir = readout_dir
        self.allow_r3_fallback = allow_r3_fallback

    def find_file(self, topology: str, model_id: str) -> Optional[Path]:
        patterns = [
            f"{topology}_canonical_{model_id}.jsonl",
            f"{topology}_{model_id}.jsonl",
            f"{topology}_{model_id}_R1.jsonl",
        ]
        for pat in patterns:
            cand = self.readout_dir / pat
            if cand.exists():
                return cand

        # In readout_R1, SEQ.jsonl is 8B, but PAR.jsonl has arm_id=llama70b!
        if topology == "SEQ" and model_id == "llama8b":
            cand = self.readout_dir / "SEQ.jsonl"
            if cand.exists():
                return cand

        if self.allow_r3_fallback:
            r3_dir = self.readout_dir.parent / "readout_R3"
            for pat in patterns:
                cand = r3_dir / pat
                if cand.exists():
                    logger.info(f"Using R3 fallback for {topology} {model_id}: {cand}")
                    return cand

        return None

    def evaluate_pair(
        self, m1: str, m2: str, topology: str
    ) -> Dict[str, KappaResult]:
        f1 = self.find_file(topology, m1)
        f2 = self.find_file(topology, m2)

        if not f1 or not f2:
            missing = []
            if not f1:
                missing.append(f"{m1} ({topology})")
            if not f2:
                missing.append(f"{m2} ({topology})")
            logger.warning(f"Skipping pair {m1} vs {m2} in {topology}: missing files {missing}")
            empty = KappaResult(0, 0.0, 0.0, float("nan"), True, "N/A (Missing File)", 0, 0, 0, 0)
            return {
                "clear_approval": empty,
                "clear_rejection": empty,
                "borderline": empty,
                "overall": empty,
            }

        d1 = load_decisions(f1)
        d2 = load_decisions(f2)

        canonical_benchmark_keys = [k for k in self.benchmark if "::" in k]
        common_keys = [k for k in canonical_benchmark_keys if k in d1 and k in d2]

        if not common_keys:
            # Fallback to prompt_id keys
            prompt_keys = [k for k in self.benchmark if "::" not in k]
            common_keys = [k for k in prompt_keys if k in d1 and k in d2]

        clear_app_1, clear_app_2 = [], []
        clear_rej_1, clear_rej_2 = [], []
        borderline_1, borderline_2 = [], []
        overall_1, overall_2 = [], []

        for k in common_keys:
            c = self.benchmark[k]
            v1 = d1[k]
            v2 = d2[k]

            overall_1.append(v1)
            overall_2.append(v2)

            if c.borderline:
                borderline_1.append(v1)
                borderline_2.append(v2)
            else:
                if c.pi_star == "approve":
                    clear_app_1.append(v1)
                    clear_app_2.append(v2)
                elif c.pi_star == "deny":
                    clear_rej_1.append(v1)
                    clear_rej_2.append(v2)

        return {
            "clear_approval": compute_concordance(clear_app_1, clear_app_2),
            "clear_rejection": compute_concordance(clear_rej_1, clear_rej_2),
            "borderline": compute_concordance(borderline_1, borderline_2),
            "overall": compute_concordance(overall_1, overall_2),
        }


def format_table_output(
    results: Dict[str, Dict[str, Dict[str, KappaResult]]],
) -> Tuple[str, str]:
    md_lines = [
        "## Pairwise Model Concordance (Cohen's Kappa & Raw Agreement P_o)",
        "",
        "| Topology | Model Pair | Clear Approval (Po, kappa) | Clear Rejection (Po, kappa) | Borderline (Po, kappa) | Overall (Po, kappa) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inter-model decision concordance (Cohen's $\kappa$ and raw agreement $P_o$) across normative credit strata. Under unanimous agreement on single-class strata, Cohen's $\kappa$ is mathematically undefined ($0/0$) and reported as NaN rather than 1.0.}",
        r"\label{tab:app_concordance}",
        r"\vspace{2pt}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"\textbf{Topology} & \textbf{Model Pair} & \textbf{Clear Approval} & \textbf{Clear Rejection} & \textbf{Borderline} & \textbf{Overall $\kappa$} \\",
        r"\midrule",
    ]

    for topo, pairs in results.items():
        for pair_label, strata in pairs.items():
            ca = strata["clear_approval"]
            cr = strata["clear_rejection"]
            bd = strata["borderline"]
            ov = strata["overall"]

            ca_disp = f"{ca.po:.1%} (`{ca.status_label}`)" if ca.n_samples > 0 else "`N/A`"
            cr_disp = f"{cr.po:.1%} (`{cr.status_label}`)" if cr.n_samples > 0 else "`N/A`"
            bd_disp = f"{bd.po:.1%} (`{bd.status_label}`)" if bd.n_samples > 0 else "`N/A`"
            ov_disp = f"{ov.po:.1%} (`{ov.status_label}`)" if ov.n_samples > 0 else "`N/A`"

            md_lines.append(
                f"| `{topo}` | {pair_label} | {ca_disp} | {cr_disp} | {bd_disp} | {ov_disp} |"
            )

            ca_str = f"$P_o={ca.po:.2f}$, {ca.status_label}" if ca.n_samples > 0 else r"\text{N/A}"
            cr_str = f"$P_o={cr.po:.2f}$, {cr.status_label}" if cr.n_samples > 0 else r"\text{N/A}"
            bd_str = (f"$\\kappa={bd.status_label}$" if not bd.is_undefined else bd.status_label) if bd.n_samples > 0 else r"\text{N/A}"
            ov_str = (f"$\\kappa={ov.status_label}$" if not ov.is_undefined else ov.status_label) if ov.n_samples > 0 else r"\text{N/A}"

            latex_lines.append(
                f"\\texttt{{{topo}}} & {pair_label} & {ca_str} & {cr_str} & {bd_str} & {ov_str} \\\\"
            )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(md_lines), "\n".join(latex_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="FairWatch V2 Model Concordance Kappa Calculator (Hardened)")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/derived/core_benchmark_v2.csv"),
        help="Path to core_benchmark_v2.csv",
    )
    parser.add_argument(
        "--readout-dir",
        type=Path,
        default=Path("results/v2/production/readout_R1"),
        help="Path to readout directory (default: results/v2/production/readout_R1)",
    )
    parser.add_argument(
        "--allow-r3-fallback",
        action="store_true",
        help="Allow fallback to readout_R3 for missing arms like 8B PAR (annotated in table)",
    )
    parser.add_argument(
        "--latex-out",
        type=Path,
        default=Path("tables/table_8_model_concordance.tex"),
        help="Optional path to output LaTeX snippet",
    )
    args = parser.parse_args()

    benchmark_path = args.benchmark.resolve()
    readout_dir = args.readout_dir.resolve()

    if not benchmark_path.exists():
        logger.error(f"Benchmark file does not exist: {benchmark_path}")
        return 1
    if not readout_dir.exists():
        logger.error(f"Readout directory does not exist: {readout_dir}")
        return 1

    evaluator = ConcordanceEvaluator(benchmark_path, readout_dir, allow_r3_fallback=args.allow_r3_fallback)

    all_results: Dict[str, Dict[str, Dict[str, KappaResult]]] = {}

    for topo in ("SEQ", "PAR"):
        topo_results: Dict[str, Dict[str, KappaResult]] = {}
        for m1, m2, label in evaluator.PAIRS:
            strata_res = evaluator.evaluate_pair(m1, m2, topo)
            topo_results[label] = strata_res
        all_results[topo] = topo_results

    md_table, latex_table = format_table_output(all_results)
    print("\n" + md_table + "\n")

    if args.latex_out:
        out_latex = args.latex_out.resolve()
        out_latex.parent.mkdir(parents=True, exist_ok=True)
        with open(out_latex, "w", encoding="utf-8") as fp:
            fp.write(latex_table)
        logger.info(f"LaTeX snippet saved to {out_latex}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
