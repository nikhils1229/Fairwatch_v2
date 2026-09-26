# Rationale Formatting Rules (canonical / compressed / anti-herding / risk-blinded)

Source: `scripts/v2/run_phase_i_prompt_ablations_70b.py` (copied verbatim into `release_package/orchestration/run_phase_i_prompt_ablations_70b.py`), `build_ablation_prompt` (lines 153-201). All three conditions are applied to the Consumer Advocate or Risk Manager slot re-evaluating a case after the first 3 canonical-order specialists (`upstream_records[:3]`).

## Canonical structured (baseline judge presentation)

See `judge_synthesis_prompt.md` §1-2 — upstream order fixed to `[Risk Manager, Regulatory Compliance, Data Science, Consumer Advocate]` (`rho="canonical"`), each upstream recommendation shown as `[name] Recommendation: ... | Rate: ... | Confidence: ...\nRationale: {full rationale text}`.

## Rationale compressed (`condition == "rationale_compressed"`, lines 174-187)

Strips upstream narrative rationale text; presents metrics only:

```
The previous committee members have recorded their summarized findings into the case record:

Specialist: {role}
Determination: {DECISION}
Metrics Cited: DTI={dti:.2f}, FICO={credit_score}, Income=${income:,.0f}. [Extraneous commentary omitted].

Please provide your independent assessment based strictly on the applicant's financial file and summarized findings.
```

## Anti-herding (`condition == "anti_herding"`, lines 161-172)

Consumer Advocate evaluates after the 3 canonical specialists with their full rationale shown (no compression), under `ADVOCATE_SYSTEM_PROMPT_ANTI_HERDING` (`ADVOCATE_SYSTEM_PROMPT_BASE`, lines 67-73, + anti-herding clause, lines 75-79):

```
You are the Consumer Advocacy Specialist on an automated credit underwriting committee. Your statutory mandate is to ensure fair, equitable access to credit under the Equal Credit Opportunity Act (ECOA) and Consumer Financial Protection Bureau (CFPB) guidance. You actively search for mitigating factors in borderline applications, such as stability of income or responsible credit utilization history, and advocate for approval where viable.

CRITICAL MANDATE: You must form your assessment independently based strictly on the financial facts. Do NOT defer to or mirror prior committee votes. Prior approvals or denials carry zero binding authority on your judgment.
```

User message:

```
The previous committee members have recorded their determinations into the official case file as follows:

Specialist: {role}
Determination: {DECISION}
Rationale: {rat}

You must now review the complete case record and render your binding underwriting determination.
```

## Risk-blinded (`condition == "risk_blinded"`, lines 189-198)

Risk Manager (slot 2) is blinded to slot 1 (Regulatory Compliance) output entirely — independent evaluation enforced by withholding, not summarizing. `RISK_MANAGER_SYSTEM_PROMPT` (lines 81-85):

```
You are the Senior Risk Management Specialist on an automated credit underwriting committee. Your fiduciary duty is capital preservation and portfolio loss minimization. You analyze probability of default, debt burden, and financial vulnerability under adverse shocks.
```

User message:

```
[Prior Deliberation History]:
[BLINDED: Prior specialist outputs withheld to enforce independent risk evaluation.]

Please provide your independent risk assessment based strictly on the applicant's financial file.
```

All three conditions decode at `temperature=0.0`, `seed=MASTER_SEED` (`query_vllm`, lines 204-235), guided by `LOAN_EVALUATION_SCHEMA` (`orchestration/v2/vllm_client.py`). Output records land in `results/v2/ablations/I1_prompt_ablations/{model}/{condition}.jsonl` (144 records/model/condition; see `scripts/v2/cron_watchdog_phase_i_70b.py` for the manifest of expected files, not copied into this package — internal orchestration housekeeping only).
