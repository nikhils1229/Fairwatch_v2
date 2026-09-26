# Pre-specified Seeds, Temperature Grid, Model Revisions

## Seeds

| Seed | Purpose | Source |
|---|---|---|
| `20260911` | Stochastic replication #1; also `MASTER_SEED` for HIER-84 Llama-70B run; permutation-variance analysis default (`--seed`) | `scripts/v2/run_phase_i_stochastic_llama8b.py:60`, `scripts/v2/run_hier84_llama70b.py:52`, `analysis/v2/compute_permutation_variance.py:214` |
| `20260912` | Stochastic replication #2 | `scripts/v2/run_phase_i_stochastic_llama8b.py:60`, `scripts/v2/cron_watchdog_d2b.py:26` |
| `20260913` | Stochastic replication #3 | `scripts/v2/run_phase_i_stochastic_llama8b.py:60`, `scripts/v2/cron_watchdog_d2b.py:27` |
| `2027` | Spotcheck gate default seed | `scripts/v2/run_spotcheck.py:7` (`def run_spotcheck(seed=2027)`) |

## Temperature grid

`T ∈ {0.0, 0.20, 0.70}` — confirmed by stochastic-decoding filenames and cron-watchdog target
lists, e.g. `results/v2/ablations/D2_precision_decoding/stochastic_qwen72b_T020_seed20260912.jsonl`
and `..._T070_seed20260912.jsonl` (`scripts/v2/cron_watchdog_d2b.py:26-30`); canonical runs use
`T=0.0` (not separately filenamed — the baseline arm).

## Decoding hyperparameter bounds

Enforced in `orchestration/harness/decoding.py::validate_sampling` (copied to
`release_package/orchestration/decoding.py`):

- `temperature`: [0.0, 2.0]
- `top_p`: [0.0, 1.0]
- `top_k`: -1 (disabled) or >= 1 (0 explicitly rejected)
- `repetition_penalty`: [0.5, 2.0]
- `max_tokens`: [1, 8192]

## Model HuggingFace IDs & exact revisions

**Verified against `scripts/v2/serve.d/*.conf` at repo HEAD `b236a0d`** (copied to
`release_package/configs/`). See `release_package/README_REPRODUCTION.md` section (b) for the
full table and a documented discrepancy against the task-supplied revision hashes for
70B/Qwen-7B/Qwen-72B (only the Llama-8B hash matched what was requested).
