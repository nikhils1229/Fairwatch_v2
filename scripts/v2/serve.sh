#!/usr/bin/env bash
set -euo pipefail
FW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FW_ROOT"

ACTION="${1:-}"
if [ "$ACTION" = "stop" ]; then
    NAME="${2:?usage: serve.sh stop <served_name>}"
    PIDFILE="run/v2/${NAME}.pid"
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        echo "Stopping $NAME (PID $PID)..."
        PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ' || true)
        if [ -n "$PGID" ]; then
            kill -TERM -"$PGID" 2>/dev/null || true
            sleep 3
            kill -KILL -"$PGID" 2>/dev/null || true
        else
            kill -TERM "$PID" 2>/dev/null || true
            sleep 3
            kill -KILL "$PID" 2>/dev/null || true
        fi
        rm -f "$PIDFILE" "run/v2/${NAME}.json"
        echo "Stopped $NAME."
    else
        echo "No pidfile for $NAME at $PIDFILE."
    fi
    exit 0
fi

bash scripts/v2/_preflight.sh

SERVED_NAME="${1:?usage: serve.sh <served_name> [wait]}"
WAIT_FLAG="${2:-}"

CONF="scripts/v2/serve.d/${SERVED_NAME}.conf"
test -r "$CONF" || { echo "FATAL: no config $CONF"; exit 2; }

set -a
source "$FW_ROOT/.env.secrets"
source "$CONF"
set +a

for v in SERVED_NAME MODEL_ID REVISION GPUS TP PORT MAX_MODEL_LEN GPU_MEM_UTIL ENGINE_SEED; do
  [ -n "${!v:-}" ] || { echo "FATAL: $v unset in $CONF"; exit 2; }
done

[ "$MAX_MODEL_LEN" = "8192" ] || { echo "FATAL: max_model_len is preregistered at 8192"; exit 2; }

# Validate 40-character commit SHA
if [[ ! "${REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "FATAL: REVISION must be a full 40-character commit SHA (got: ${REVISION})" >&2
    exit 2
fi

# Validate GPU constraint: max 3 GPUs allowed
IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [ "${#GPU_ARR[@]}" -gt 3 ]; then
    echo "FATAL: Requested ${#GPU_ARR[@]} GPUs (${GPUS}), but max 3 GPUs allowed!" >&2
    exit 2
fi

# Hardware boundary assertion for multi-tenant cluster safety:
# When serving secondary or control models like qwen72b, GPUs 0-3 are protected tenant devices.
if true; then  # Strict cluster boundary for all models
    for g in "${GPU_ARR[@]}"; do
        if [ "$g" -lt 3 ]; then
            echo "FATAL CRITICAL BREACH: GPU $g is within protected tenant range [0, 2]!" >&2
            exit 99
        fi
    done
fi

# Port check
if ss -ltn "sport = :$PORT" | grep -q LISTEN ; then
    echo "FATAL: port $PORT occupied"; exit 2
fi

export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

QFLAG=()
[ -n "${QUANT:-}" ] && QFLAG=(--quantization "$QUANT")

echo "Launching ${SERVED_NAME} (${MODEL_ID}) on GPUs ${GPUS} (TP=${TP}) port ${PORT}..."

MAX_SEQS="${MAX_NUM_SEQS_RESOLVED:-${MAX_NUM_SEQS:-64}}"
if [ "$MAX_SEQS" = "AUTO" ]; then
    MAX_SEQS=64
fi

EXTRA_ARGS=()
if [ "$TP" -gt 1 ]; then
    EXTRA_ARGS+=(--disable-custom-all-reduce)
fi

setsid env CUDA_VISIBLE_DEVICES="$GPUS"     NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1     "$FW_ROOT/.venv_fw/bin/python"     -m vllm.entrypoints.openai.api_server     --model "$MODEL_ID" --revision "$REVISION" --served-model-name "$SERVED_NAME"     --tensor-parallel-size "$TP" "${QFLAG[@]}" "${EXTRA_ARGS[@]}" --dtype "${DTYPE:-auto}"     --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_SEQS"     --gpu-memory-utilization "$GPU_MEM_UTIL"     --host 127.0.0.1 --port "$PORT" --seed "$ENGINE_SEED"     --enable-prefix-caching --no-enable-log-requests     >> "logs/v2/serve_${SERVED_NAME}.log" 2>&1 &

PID=$!
echo $PID > "run/v2/${SERVED_NAME}.pid"
PGID=$(ps -o pgid= -p "$PID" 2>/dev/null | tr -d ' ' || echo "$PID")
GIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
STARTED_AT=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

"$FW_ROOT/.venv_fw/bin/python" -c "
import json, pathlib
run_meta = {
    'pid': $PID,
    'pgid': $PGID,
    'host': '127.0.0.1',
    'port': $PORT,
    'gpus': '$GPUS',
    'model_id': '$MODEL_ID',
    'revision': '$REVISION',
    'served_name': '$SERVED_NAME',
    'git_sha': '$GIT_SHA',
    'started_at': '$STARTED_AT',
    'max_model_len': $MAX_MODEL_LEN,
    'max_num_seqs': $MAX_SEQS,
    'tp': $TP
}
with open('run/v2/${SERVED_NAME}.json', 'w') as f:
    json.dump(run_meta, f, indent=2)
"

echo "Server process $PID launched (PGID $PGID). Logs: logs/v2/serve_${SERVED_NAME}.log"

if [ "$WAIT_FLAG" = "wait" ] || [ "$WAIT_FLAG" = "--wait" ]; then
    echo "Waiting for ${SERVED_NAME} readiness on port ${PORT} (timeout 900s)..."
    READY=0
    for i in $(seq 1 180); do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo "FATAL: Process $PID exited unexpectedly. Traceback:"
            tail -n 40 "logs/v2/serve_${SERVED_NAME}.log"
            exit 2
        fi
        if curl -s "http://127.0.0.1:${PORT}/v1/models" | grep -q "$SERVED_NAME" ; then
            READY=1
            break
        fi
        sleep 5
    done
    if [ "$READY" -eq 0 ]; then
        echo "FATAL: Timed out waiting for ${SERVED_NAME} on port ${PORT}"
        kill -TERM -"$PGID" 2>/dev/null || true
        exit 2
    fi
    echo "${SERVED_NAME} is READY on http://127.0.0.1:${PORT}"
fi
