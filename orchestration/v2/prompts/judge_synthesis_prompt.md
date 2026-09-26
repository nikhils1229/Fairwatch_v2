# Judge / Synthesis Prompt (R2 readout)

Source: `orchestration/v2/run_judge_replay.py` (copied verbatim into `release_package/orchestration/run_judge_replay.py`), functions `format_judge_input` (lines 35–80) and `evaluate_single_case` (readout `R2` branch).

## 1. Judge prompt template (exact, `run_judge_replay.py:72-77`)

```
You are the Senior Presiding Underwriter. Synthesize a final business decision based on the applicant data and advisor recommendations.

LOAN APPLICATION:
{prompt_text}

ADVISOR RECOMMENDATIONS:
{recommendations_text}

Synthesize these inputs into a final credit decision. Return valid JSON only.
```

`{prompt_text}` = the original loan-application dossier text. `{recommendations_text}` = the four upstream agent recommendation snippets joined by `\n\n`, each formatted as (`run_judge_replay.py:64`):

```
[{name}] Recommendation: {dec} | Rate: {rate_str} | Confidence: {conf_str} ({conf_lvl})
Rationale: {rationale}
```

## 2. Ordering / labeling parameters (rho, label_mode — `format_judge_input`, lines 35-53)

- `rho == "canonical"`: upstream recommendations are sorted into fixed order `["Risk Manager", "Regulatory Compliance", "Data Science", "Consumer Advocate"]` before presentation to the judge (matched by substring on `agent_name`, case-insensitive). Any other `rho` value leaves upstream order as stored (permutation order, used for the order-sensitivity sweeps).
- `label_mode in ("labels_off", "anonymous", "blinded")`: agent identity is replaced with `Advisor_A`/`Advisor_B`/... in presentation order (risk-blinded / anonymized presentation). Otherwise the real `agent_name` is shown.

## 3. Structured JSON output schema (judge output)

`BUSINESS_SYNTHESIS_SCHEMA` in `orchestration/v2/vllm_client.py` (copied to `release_package/orchestration/vllm_client.py`) — same field set as the upstream `LOAN_EVALUATION_SCHEMA` plus a nested `reasoning` object (`approval_decision_reason`, `approval_type_reason`, `interest_rate_reason`, `confidence_reason`). `approval_decision` enum is strictly `["approve", "deny"]`, enforced via guided/structured decoding (`response_format: {"type": "json_schema", ...}`), `additionalProperties: false`.

## 4. Alternative readouts (not judge-LLM based, for comparison)

- **R1 (majority consensus)**: `orchestration.harness.readout.R1_majority` — deterministic vote count over the four upstream `approval_decision` values, no LLM call.
- **R3 (terminal agent)**: reuses `BusinessDecisionAgent` (`agents/business_decision_agent.py`) with the same `normalize_decision` mapping as upstream agents; no separate judge prompt.
