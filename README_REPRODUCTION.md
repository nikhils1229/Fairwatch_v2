# FairWatch — Reproduction Guide

All commands assume `cd` into repo root, `FW_ROOT` set to that path (`export FW_ROOT=$(pwd)`).
Every command below is copied from the actual script it invokes (argparse defaults / shebang) —
no parameter here is invented.

## a) Environment setup

```bash
python3 -m venv .venv_fw
source .venv_fw/bin/activate
pip install -r requirements.txt          # pins vllm>=0.4.0 (requirements.txt:1)
```

## b) Serving models (`scripts/v2/serve.sh`, `scripts/v2/serve.d/*.conf`)

Each `.conf` pins `MODEL_ID` + exact HF `REVISION` (see `configs/*.conf` in this package).
Launch a model:

```bash
bash scripts/v2/serve.sh llama8b wait     # blocks until healthy; see serve.d/llama8b.conf
bash scripts/v2/serve.sh llama70b wait
bash scripts/v2/serve.sh qwen7b wait
bash scripts/v2/serve.sh qwen72b wait
bash scripts/v2/serve.sh <served_name> stop   # tear down
```

Underlying vLLM invocation (`scripts/v2/serve.sh`, engine launch line) uses, per config:
`--model $MODEL_ID --revision $REVISION --tensor-parallel-size $TP --seed $ENGINE_SEED
--enable-prefix-caching --no-enable-log-requests`.

Verified model IDs / revisions on this cluster (`scripts/v2/serve.d/*.conf`, checked 2026-09-23):

| served_name | MODEL_ID | REVISION | TP |
|---|---|---|---|
| llama3b | meta-llama/Llama-3.2-3B-Instruct | `0cb88a4f764b7a12671c53f0838cd831a0843b95` | 1 |
| llama8b | meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | 1 |
| llama70b | hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 | `2123003760781134cfc31124aa6560a45b491fdf` | 2 |
| qwen3b | Qwen/Qwen2.5-3B-Instruct | `aa8e72537993ba99e69dfaafa59ed015b17504d1` | 1 |
| qwen7b | Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | 1 |
| qwen72b | Qwen/Qwen2.5-72B-Instruct-AWQ | `698703eae6604af048a3d2f509995dc302088217` | 2 |

**NOTE**: the task brief for this release step specified different revision hashes and
un-quantized model IDs for 70B/Qwen-7B/Qwen-72B. Those do not match the repo's tracked
`serve.d/*.conf` files (only the llama8b hash matched). The table above is the verified
value read directly from `scripts/v2/serve.d/*.conf` at HEAD `b236a0d`; see `NOTES.md`
("PHASE J — Release Package") for the discrepancy record.

Sampling policy bounds are enforced in `orchestration/harness/decoding.py`
(`SamplingPolicy`, `validate_sampling`): `temperature ∈ [0.0, 2.0]`, `top_p ∈ [0.0, 1.0]`,
`repetition_penalty ∈ [0.5, 2.0]`, `max_tokens ∈ [1, 8192]`. Manuscript temperature grid
(T ∈ {0.0, 0.20, 0.70}) and pre-specified seeds (20260911, 20260912, 20260913, 2027) are
consumed as CLI args by the run/analysis scripts below — see each script's `--seed`/`--temp`
default in its `argparse` block for the exact value used per run.

## c) Running evaluations and permutation sweeps

Judge replay (R1/R2/R3 readouts) over frozen message-sets — `orchestration/v2/run_judge_replay.py`:

```bash
python3 -m orchestration.v2.run_judge_replay \
  --message-sets <path/to/message_set_dir> \
  --readout R2 \
  --rho canonical \
  --label-mode labels_off \
  --model llama8b --port 8002 --policy primary --guided true \
  --concurrency 8 \
  --out results/v2/production/readout_R2
```

(`--readout` choices: `R1`, `R2`, `R3`. `--label-mode` choices: `labels_off`, `named`,
`anonymous`. Defaults per `orchestration/v2/run_judge_replay.py` argparse block.)

## d) Running analysis scripts to reproduce tables/figures

```bash
python3 analysis/v2/compute_permutation_variance.py \
  --input results/v2/production/readout_R1/SEQ_full24_llama8b.jsonl \
  --num-permutations 100000 --seed 20260911 \
  --output results/v2/permutation_variance_report.json

python3 analysis/v2/analyze_borderline31_sweep.py
python3 analysis/v2/compute_cluster_robust_and_mixed_effects.py
python3 analysis/v2/compute_model_concordance_kappa.py
```

Each script's exact CLI surface (flags, defaults) is defined in its own `argparse` block;
run `python3 <script>.py --help` for the authoritative list rather than trusting a copy here.

## Raw data prerequisite (important)

`results/v2/` (raw generation/readout `.jsonl`, ~3.9 GB) is listed in `.gitignore`
(`results/**`) and is **not part of git history** — confirmed by `git log --oneline -1 --
results/v2/production/readout_R1/SEQ_full24_llama8b.jsonl` returning no commits, while the
file exists on disk. A `git clone` alone will NOT reproduce any table. The raw `results/v2/`
tree must be obtained out-of-band (data release artifact / direct copy) and placed at
`$FW_ROOT/results/v2/` before running section (d) above. See
`docs/reproduction/fresh_clone_reproduce.log` for a worked example.
