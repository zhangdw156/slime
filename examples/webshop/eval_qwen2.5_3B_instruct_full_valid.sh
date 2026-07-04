#!/bin/bash

# Eval-only launcher for the full WebShop held-out validation pool.
# It writes a separate 500-goal full-valid JSONL schedule, then starts slime
# with --num-rollout 0 so no training rollout/update is run.

if [[ "${WEBSHOP_FORCE_CLEANUP:-0}" == "1" ]]; then
   echo "WEBSHOP_FORCE_CLEANUP=1: stopping local Ray/SGLang/Python processes before launch."
   pkill -9 sglang || true
   sleep 3
   ray stop --force || true
   pkill -9 ray || true
   pkill -9 python || true
   sleep 3
   pkill -9 ray || true
   pkill -9 python || true
fi

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

MODEL_ROOT=${MODEL_ROOT:-/root/Qwen2.5-3B-Instruct}
SLIME_CKPT=${SLIME_CKPT:-/root/Qwen2.5-3B-Instruct_webshop_grpo_slime}
MCORE_CKPT=${MCORE_CKPT:-}
CKPT_STEP=${CKPT_STEP:-}

WEBSHOP_FULL_EVAL_TASK_DIR=${WEBSHOP_FULL_EVAL_TASK_DIR:-/root/slime-webshop-full-eval}
FULL_VALID_DATA="${WEBSHOP_FULL_EVAL_TASK_DIR}/valid_full.jsonl"
export WEBSHOP_SERVICE_URL=${WEBSHOP_SERVICE_URL:-http://127.0.0.1:3001}
export WEBSHOP_HISTORY_LENGTH=${WEBSHOP_HISTORY_LENGTH:-4}

require_path() {
   local path="$1"
   local desc="$2"
   if [[ ! -e "${path}" ]]; then
      echo "ERROR: missing ${desc}: ${path}" >&2
      exit 2
   fi
}

check_webshop_service() {
   python3 - <<'PY'
import json
import os
import urllib.request

url = os.environ["WEBSHOP_SERVICE_URL"].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 - operator-provided service URL.
        payload = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    raise SystemExit(f"ERROR: failed to reach WebShop service {url}: {exc}")
if not payload.get("ok"):
    raise SystemExit(f"ERROR: unhealthy WebShop service response: {payload}")
goals = int(payload.get("goal_count", payload.get("goals", 0)))
if goals <= 0:
    raise SystemExit(f"ERROR: WebShop service has no goals: {payload}")
expected = 6910
if goals != expected:
    raise SystemExit(f"ERROR: expected {expected} WebShop small synthetic goals, got {goals}: {payload}")
print(json.dumps({"webshop_service": url, "goals": goals, "sessions": payload.get("sessions")}, indent=2))
PY
}

require_path "${MODEL_ROOT}" "HF model root"
require_path "${SLIME_CKPT}" "slime checkpoint"
if [[ -n "${MCORE_CKPT}" ]]; then
   require_path "${MCORE_CKPT}" "optional Megatron torch_dist checkpoint"
fi
if [[ -n "${CKPT_STEP}" ]]; then
   printf -v CKPT_ITER_DIR 'iter_%07d' "${CKPT_STEP}"
   if [[ ! -d "${SLIME_CKPT}/${CKPT_ITER_DIR}" ]]; then
      echo "Requested CKPT_STEP=${CKPT_STEP}, but ${SLIME_CKPT}/${CKPT_ITER_DIR} does not exist." >&2
      exit 1
   fi
else
   CKPT_ITER_DIR=latest
   if [[ ! -f "${SLIME_CKPT}/latest_checkpointed_iteration.txt" ]]; then
      echo "CKPT_STEP is not set and latest_checkpointed_iteration.txt is missing under ${SLIME_CKPT}." >&2
      exit 1
   fi
fi
check_webshop_service

mkdir -p "${WEBSHOP_FULL_EVAL_TASK_DIR}"
python3 "${SCRIPT_DIR}/prepare_webshop_data.py" \
   --service-url "${WEBSHOP_SERVICE_URL}" \
   --train-start 500 \
   --valid-size 500 \
   --output-dir "${WEBSHOP_FULL_EVAL_TASK_DIR}"
mv "${WEBSHOP_FULL_EVAL_TASK_DIR}/valid.jsonl" "${FULL_VALID_DATA}"

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
NUM_GPUS=${NUM_GPUS:-2}
TP_SIZE=${TP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-32768}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-8192}
CKPT_LABEL=${CKPT_STEP:-latest}
EVAL_TAG=${EVAL_TAG:-full-valid-${CKPT_LABEL}}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/"
   --load "${SLIME_CKPT}/"
   --no-load-optim
   --no-load-rng
   --finetune
)
if [[ -n "${MCORE_CKPT}" ]]; then
   CKPT_ARGS+=(--ref-load "${MCORE_CKPT}/")
fi
if [[ -n "${CKPT_STEP}" ]]; then
   CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")
fi

ROLLOUT_ARGS=(
   --input-key text
   --metadata-key metadata
   --num-rollout 0
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data valid_full "${FULL_VALID_DATA}"
   --n-samples-per-eval-prompt 1
   --eval-temperature 0.4
   --eval-top-p 1.0
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" == "1" ]]; then
   WANDB_ARGS+=(
      --use-wandb
      --wandb-project "${WANDB_PROJECT:-slime-webshop}"
      --wandb-group "${WANDB_GROUP:-qwen2.5-3B-instruct-webshop-eval-${EVAL_TAG}}"
      --disable-wandb-random-suffix
   )
   if [[ -n "${WANDB_MODE:-}" ]]; then
      WANDB_ARGS+=(--wandb-mode "${WANDB_MODE}")
   fi
   if [[ -n "${WANDB_KEY:-}" ]]; then
      WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
   fi
   if [[ -n "${WANDB_HOST:-}" ]]; then
      WANDB_ARGS+=(--wandb-host "${WANDB_HOST}")
   fi
   if [[ -n "${WANDB_TEAM:-}" ]]; then
      WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
   fi
   if [[ -n "${WANDB_DIR:-}" ]]; then
      WANDB_ARGS+=(--wandb-dir "${WANDB_DIR}")
   fi
fi

SWANLAB_ARGS=()
if [[ "${USE_SWANLAB:-1}" == "1" ]]; then
   SWANLAB_ARGS+=(
      --use-swanlab
      --swanlab-mode "${SWANLAB_MODE:-cloud}"
      --swanlab-project "${SWANLAB_PROJECT:-slime-webshop}"
      --swanlab-group "${SWANLAB_GROUP:-qwen2.5-3B-instruct-eval}"
      --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-qwen2.5-3B-instruct-webshop-eval-${EVAL_TAG}}"
      --disable-swanlab-random-suffix
   )
   if [[ -n "${SWANLAB_API_KEY:-}" ]]; then
      SWANLAB_ARGS+=(--swanlab-key "${SWANLAB_API_KEY}")
   fi
   if [[ -n "${SWANLAB_WORKSPACE:-}" ]]; then
      SWANLAB_ARGS+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
   fi
   if [[ -n "${SWANLAB_DIR:-}" ]]; then
      SWANLAB_ARGS+=(--swanlab-dir "${SWANLAB_DIR}")
   fi
   if [[ -n "${SWANLAB_HOST:-}" ]]; then
      SWANLAB_ARGS+=(--swanlab-host "${SWANLAB_HOST}")
   fi
   if [[ -n "${SWANLAB_WEB_HOST:-}" ]]; then
      SWANLAB_ARGS+=(--swanlab-web-host "${SWANLAB_WEB_HOST}")
   fi
   if [[ -n "${SWANLAB_OPEN_METRICS_INTERVAL:-}" ]]; then
      SWANLAB_ARGS+=(--swanlab-open-metrics-interval "${SWANLAB_OPEN_METRICS_INTERVAL}")
   fi
fi

TENSORBOARD_ARGS=()
if [[ "${USE_TENSORBOARD:-0}" == "1" ]]; then
   TENSORBOARD_ARGS+=(
      --use-tensorboard
      --tb-project-name "${TB_PROJECT_NAME:-slime-webshop}"
      --tb-experiment-name "${TB_EXPERIMENT_NAME:-qwen2.5-3B-instruct-webshop-eval-${EVAL_TAG}}"
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.6}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path generate_with_webshop.generate
   --custom-eval-rollout-log-function-path generate_with_webshop.log_webshop_eval_rollout
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/root/shared/ray_temp}
mkdir -p "${RAY_TEMP_DIR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" --temp-dir "${RAY_TEMP_DIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WEBSHOP_SERVICE_URL\": \"${WEBSHOP_SERVICE_URL}\",
    \"WEBSHOP_HISTORY_LENGTH\": \"${WEBSHOP_HISTORY_LENGTH}\"
  }
}"

ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   --rollout-num-gpus "${NUM_GPUS}" \
   --colocate \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${DISTRIBUTED_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SWANLAB_ARGS[@]}" \
   "${TENSORBOARD_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
