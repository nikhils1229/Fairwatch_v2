#!/usr/bin/env python3
"""
FairWatch V2 — Comprehensive Data & Run Integrity Forensic Auditor
Audits all JSON/JSONL result files across results/v2/ to guarantee:
  1. Zero corrupt, truncated, or malformed JSON records.
  2. Zero null, missing, or non-binary decisions (must be strictly 'approve' or 'deny').
  3. Zero unhandled API errors, tracebacks, or parse failures.
  4. Detection of degenerate failure modes (e.g. 100% denials only, 100% approvals, or frozen outputs).
Outputs an exhaustive report table with exact file paths and statistics.
"""
from __future__ import annotations
import json, os, sys
from typing import Any
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT_DIR / "results/v2"

TARGET_DIRS = [
    RESULTS_DIR / "production/readout_R1",
    RESULTS_DIR / "production/readout_R2",
    RESULTS_DIR / "production/readout_R3",
    RESULTS_DIR / "solo",
    RESULTS_DIR / "solo/D3_solo_executive_judge/llama70b",
    RESULTS_DIR / "solo/D3_solo_executive_judge/qwen72b",
    RESULTS_DIR / "ablations/D1_cascade_prompt_ablation/llama70b",
    RESULTS_DIR / "ablations/D1_cascade_prompt_ablation/qwen72b",
    RESULTS_DIR / "ablations/D2_precision_decoding",
    RESULTS_DIR / "ablations/wrapper_ablation",
]


def extract_decision(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    # Common decision keys across topologies
    for k in ["approval_decision", "decision", "final_decision", "verdict", "executive_decision"]:
        if k in data and data[k] is not None:
            v = str(data[k]).strip().lower()
            if v in ["approve", "approved", "1", "true"]: return "approve"
            if v in ["deny", "denied", "reject", "rejected", "0", "false"]: return "deny"
    # Check nested fields
    if "data" in data and isinstance(data["data"], dict):
        return extract_decision(data["data"])
    if "result" in data and isinstance(data["result"], dict):
        return extract_decision(data["result"])
    return None


def audit_jsonl_file(file_path: Path) -> dict:
    total_lines = 0
    corrupt_lines = 0
    missing_decisions = 0
    approvals = 0
    denials = 0
    error_markers = 0

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            total_lines += 1

            # Check JSON parse
            try:
                data = json.loads(line_str)
            except Exception:
                corrupt_lines += 1
                continue

            # Check for error statuses or tracebacks
            raw_text = line_str.lower()
            if any(err in raw_text for err in ["traceback (most recent call last)", "internalservererror", "cuda out of memory", "parse_unparseable"]):
                error_markers += 1

            if isinstance(data, dict):
                if data.get("parse_status") in ["PARSE_ERROR", "PARSE_UNPARSEABLE"]:
                    error_markers += 1

            dec = extract_decision(data)
            if dec == "approve":
                approvals += 1
            elif dec == "deny":
                denials += 1
            else:
                missing_decisions += 1

    app_rate = (approvals / total_lines) if total_lines > 0 else 0.0
    
    # Degeneracy check: flag if completely 100% denials or 100% approvals on large sample (N >= 40)
    is_degenerate = (total_lines >= 40) and (approvals == 0 or denials == 0)
    
    status = "PASS"
    issues = []
    if corrupt_lines > 0:
        status = "FAIL"
        issues.append(f"{corrupt_lines} corrupt JSON lines")
    if missing_decisions > 0:
        status = "FAIL"
        issues.append(f"{missing_decisions} missing decisions")
    if error_markers > 0:
        status = "FAIL"
        issues.append(f"{error_markers} error markers")
    if is_degenerate:
        status = "DEGENERATE"
        issues.append(f"Degenerate distribution ({approvals} Approve / {denials} Deny)")

    return {
        "file": str(file_path.relative_to(ROOT_DIR)),
        "total_records": total_lines,
        "corrupt_lines": corrupt_lines,
        "missing_decisions": missing_decisions,
        "approvals": approvals,
        "denials": denials,
        "approval_rate": round(app_rate, 4),
        "error_markers": error_markers,
        "status": status,
        "issues": "; ".join(issues) if issues else "None"
    }


def main():
    print(f"Scanning target directories for result files in {RESULTS_DIR}...")
    jsonl_files = []
    for d in TARGET_DIRS:
        if d.exists():
            for f in sorted(d.glob("*.jsonl")):
                if not f.is_symlink() and not f.name.endswith(".bak") and "bak" not in f.name:
                    jsonl_files.append(f)

    print(f"Found {len(jsonl_files)} primary JSONL dataset files to audit.\n")
    results = []
    for f in jsonl_files:
        res = audit_jsonl_file(f)
        results.append(res)

    # Print markdown table
    print("| Primary Result File | Records | Approvals | Denials | Appr Rate | Corrupt | Missing | Errors | Status |")
    print("|---|---|---|---|---|---|---|---|---|")
    total_records = 0
    total_corrupt = 0
    total_missing = 0
    total_errors = 0
    
    fail_count = 0
    degenerate_count = 0

    for r in results:
        total_records += r["total_records"]
        total_corrupt += r["corrupt_lines"]
        total_missing += r["missing_decisions"]
        total_errors += r["error_markers"]
        if r["status"] == "FAIL":
            fail_count += 1
        elif r["status"] == "DEGENERATE":
            degenerate_count += 1
            
        print(f"| `{r['file']}` | {r['total_records']} | {r['approvals']} | {r['denials']} | {r['approval_rate']*100:.1f}% | {r['corrupt_lines']} | {r['missing_decisions']} | {r['error_markers']} | **{r['status']}** |")

    print("\n=== AUDIT SUMMARY ===")
    print(f"Total Datasets Audited: {len(results)}")
    print(f"Total Decisions Inspected: {total_records:,}")
    print(f"Total Corrupt JSON Records: {total_corrupt}")
    print(f"Total Missing/Null Decisions: {total_missing}")
    print(f"Total Error Statuses / Tracebacks: {total_errors}")
    print(f"Failed Files: {fail_count}")
    print(f"Degenerate Files (100% Deny or 100% Approve): {degenerate_count}")

    out_json = ROOT_DIR / "analysis/v2/data_integrity_audit_report.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport written to: {out_json}")


if __name__ == "__main__":
    main()
