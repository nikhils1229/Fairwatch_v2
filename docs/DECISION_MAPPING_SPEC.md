# Output-to-Decision Mapping Specification (F4)

Source of truth: `agents/base_agent.py` (`normalize_decision`, `BaseAgent._fix_null_values`,
`BaseAgent.evaluate_loan_application`). Copied verbatim into `release_package/agents/base_agent.py`.

## 1. Output contract (JSON schema, enforced by system prompt)

Built in `BaseAgent._build_system_prompt()`. Required fields, all mandatory, no nulls:

```
approval_decision      : "approve" | "deny"
approval_type          : "STANDARD_TERMS" | "CONDITIONAL_APPROVAL" | "DENIAL"
interest_rate           : float, bounds from template["validation_rules"]["interest_rate_range"]
confidence_probability  : int 0-100
confidence_level        : "low" | "medium" | "high"
approval_decision_reason / approval_type_reason / interest_rate_reason / confidence_reason : string
```

## 2. Decision regex / enum mapping (`normalize_decision`, `agents/base_agent.py:51-66`)

Raw string is lowercased, `.strip('.,;:!"\'')`, underscores replaced with spaces, then matched
against an exact enum set (no regex, no substring/fuzzy matching):

- `approve` set: `{approve, approved, approval, i approve, recommend approval, decision: approve}`
- `deny` set: `{deny, denied, denial, decline, declined, reject, rejected, i deny, recommend denial, decision: deny}`
- Anything else (prose, ambiguous, empty) → `None`. **`None` is never coerced to `deny`.**

## 3. Rejection / retry logic (`BaseAgent.evaluate_loan_application`, lines 396-485)

1. Attempt 1: generate, strip markdown fences (`_clean_json`), `json.loads` with
   `parse_float=Decimal, parse_int=Decimal, parse_constant=_reject_constant` (rejects
   `NaN`/`Infinity`/`-Infinity` tokens outright).
2. On `JSONDecodeError`/`ValueError`: Attempt 2 — re-prompt with the literal error-feedback
   retry template (`retry_prompt`, lines 440-448), `parsed['retry_reason'] = 'json_syntax_error'`.
3. If both attempts fail to parse, or the client call itself fails: `_create_error_response`
   is returned — `approval_decision=None`, `parse_status="error"`, full raw text preserved
   under `raw_excerpt`/`raw_response`.

## 4. Fallback prohibition

- No default/fallback decision is ever substituted for an unparseable generation.
  `_create_error_response` docstring (`agents/base_agent.py:487-497`) records that the prior
  behavior — returning `approval_decision='deny'` with a sentinel "System Error:" string — was
  found retroactively in 103 stored result files and has been removed; a decision is `None`
  and carries `parse_status` explaining why, never counted as a denial.
- `parse_status` enum (`orchestration/harness/parse_status.py`, imported at `base_agent.py:48`):
  `PARSE_OK`, `PARSE_UNPARSEABLE`, `PARSE_ERROR`, `PARSE_REPAIRED_NULL`, `PARSE_REPAIRED_FUZZY`,
  `PARSE_FALLBACK`. Only `PARSE_OK` records are eligible for approval-rate denominators; the
  rule is stated directly in the module comment at `base_agent.py:42-47`.

## 5. Judge/synthesis-stage mapping (R2 readout)

`orchestration/v2/run_judge_replay.py:130` re-applies the identical `normalize_decision()`
function to the judge's `approval_decision` output — no separate mapping logic exists for the
synthesis stage. R1 (majority) uses `orchestration/harness/readout.R1_majority`, which is not
copied into this package (not requested); R3 (terminal agent) reuses the same `normalize_decision`
path through `BusinessDecisionAgent`.
