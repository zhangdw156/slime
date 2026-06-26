#!/bin/bash

# Optional destructive cleanup for dedicated nodes only. It is disabled by
# default because broad pkill/ray stop can kill unrelated jobs on shared servers.
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

set -ex

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: ${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SLIME_ROOT}/scripts/models/qwen2.5-3B.sh"

MODEL_ROOT=${MODEL_ROOT:-/root/Qwen2.5-3B-Instruct}
MCORE_CKPT=${MCORE_CKPT:-/root/Qwen2.5-3B-Instruct_torch_dist}
SLIME_CKPT=${SLIME_CKPT:-/root/Qwen2.5-3B-Instruct_webshop_grpo_slime}
WEBSHOP_TASK_DIR=${WEBSHOP_TASK_DIR:-/root/slime-webshop}
export WEBSHOP_SERVICE_URL=${WEBSHOP_SERVICE_URL:-http://127.0.0.1:3001}
export WEBSHOP_MAX_STEPS=${WEBSHOP_MAX_STEPS:-30}
export WEBSHOP_HISTORY_LENGTH=${WEBSHOP_HISTORY_LENGTH:-4}
export WEBSHOP_STEP_MAX_TOKENS=${WEBSHOP_STEP_MAX_TOKENS:-256}
export WEBSHOP_MAX_PROMPT_CHARS=${WEBSHOP_MAX_PROMPT_CHARS:-12000}
export WEBSHOP_INVALID_ACTION_PENALTY=${WEBSHOP_INVALID_ACTION_PENALTY:-0.0}
export WEBSHOP_CLOSE_SESSION_ON_DONE=${WEBSHOP_CLOSE_SESSION_ON_DONE:-1}
export WEBSHOP_HTTP_RETRIES=${WEBSHOP_HTTP_RETRIES:-10}
export WEBSHOP_OBSERVATION_MODE=${WEBSHOP_OBSERVATION_MODE:-text}

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
import sys
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
print(json.dumps({"webshop_service": url, "goals": goals, "sessions": payload.get("sessions")}, indent=2))
PY
}

require_path "${MODEL_ROOT}" "HF model root"
require_path "${MCORE_CKPT}" "Megatron torch_dist checkpoint"
require_path "${WEBSHOP_TASK_DIR}/train.jsonl" "WebShop train goal index"
require_path "${WEBSHOP_TASK_DIR}/valid_seen.jsonl" "WebShop valid_seen goal index"
require_path "${WEBSHOP_TASK_DIR}/valid_unseen.jsonl" "WebShop valid_unseen goal index"
check_webshop_service
mkdir -p "${SLIME_CKPT}"

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
NUM_GPUS=${NUM_GPUS:-4}
TP_SIZE=${TP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-12288}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/"
   --ref-load "${MCORE_CKPT}/"
   --load "${SLIME_CKPT}/"
   --save "${SLIME_CKPT}/"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${WEBSHOP_TASK_DIR}/train.jsonl"
   --input-key text
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${WEBSHOP_STEP_MAX_TOKENS}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --dynamic-sampling-filter-path generate_with_webshop.check_episode_reward_nonzero_std
   --custom-reward-post-process-path generate_with_webshop.grpo_normalize_webshop_steps
   --custom-rollout-log-function-path generate_with_webshop.log_webshop_rollout
   --balance-data
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL}"
   --eval-prompt-data valid_seen "${WEBSHOP_TASK_DIR}/valid_seen.jsonl" valid_unseen "${WEBSHOP_TASK_DIR}/valid_unseen.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len "${WEBSHOP_STEP_MAX_TOKENS}"
   --eval-top-k 1
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
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.01
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-webshop
   # --wandb-group qwen2.5-3B-instruct
   # --wandb-key ${WANDB_KEY}
)

SWANLAB_ARGS=(
   --use-swanlab
   --swanlab-mode "${SWANLAB_MODE:-cloud}"
   --swanlab-project "${SWANLAB_PROJECT:-slime-webshop}"
   --swanlab-group "${SWANLAB_GROUP:-qwen2.5-3B-instruct}"
   --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-qwen2.5-3B-instruct-webshop-grpo}"
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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   # WebShop is an external environment; reduce this if the service becomes the bottleneck.
   # --sglang-server-concurrency 32
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
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/root/shared/ray_temp_webshop}
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
mkdir -p "${RAY_TEMP_DIR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --port "${RAY_PORT}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" --temp-dir "${RAY_TEMP_DIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WEBSHOP_SERVICE_URL\": \"${WEBSHOP_SERVICE_URL}\",
    \"WEBSHOP_MAX_STEPS\": \"${WEBSHOP_MAX_STEPS}\",
    \"WEBSHOP_HISTORY_LENGTH\": \"${WEBSHOP_HISTORY_LENGTH}\",
    \"WEBSHOP_STEP_MAX_TOKENS\": \"${WEBSHOP_STEP_MAX_TOKENS}\",
    \"WEBSHOP_MAX_PROMPT_CHARS\": \"${WEBSHOP_MAX_PROMPT_CHARS}\",
    \"WEBSHOP_INVALID_ACTION_PENALTY\": \"${WEBSHOP_INVALID_ACTION_PENALTY}\",
    \"WEBSHOP_CLOSE_SESSION_ON_DONE\": \"${WEBSHOP_CLOSE_SESSION_ON_DONE}\",
    \"WEBSHOP_HTTP_RETRIES\": \"${WEBSHOP_HTTP_RETRIES}\",
    \"WEBSHOP_OBSERVATION_MODE\": \"${WEBSHOP_OBSERVATION_MODE}\"
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
   "${GRPO_ARGS[@]}" \
   "${DISTRIBUTED_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SWANLAB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
