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
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-3B.sh"

MODEL_ROOT=${MODEL_ROOT:-/root/Qwen2.5-3B-Instruct}
MCORE_CKPT=${MCORE_CKPT:-/root/Qwen2.5-3B-Instruct_torch_dist}
SLIME_CKPT=${SLIME_CKPT:-/root/Qwen2.5-3B-Instruct_webshop_grpo_slime}
WEBSHOP_TASK_DIR=${WEBSHOP_TASK_DIR:-/root/slime-webshop}
export WEBSHOP_SERVICE_URL=${WEBSHOP_SERVICE_URL:-http://127.0.0.1:3001}
export WEBSHOP_HISTORY_LENGTH=${WEBSHOP_HISTORY_LENGTH:-4}
export WEBSHOP_REWARD_MODE=${WEBSHOP_REWARD_MODE:-dense}

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
num_products = payload.get("num_products")
if num_products not in (1000, "1000"):
    raise SystemExit(f"ERROR: expected WebShop service num_products=1000 for small synthetic goals, got {num_products}: {payload}")
print(json.dumps({"webshop_service": url, "goals": goals, "num_products": num_products, "sessions": payload.get("sessions")}, indent=2))
PY
}

require_path "${MODEL_ROOT}" "HF model root"
require_path "${MCORE_CKPT}" "Megatron torch_dist checkpoint"
require_path "${WEBSHOP_TASK_DIR}/train.jsonl" "WebShop train goal index"
require_path "${WEBSHOP_TASK_DIR}/valid.jsonl" "WebShop validation goal index"
check_webshop_service
mkdir -p "${SLIME_CKPT}"

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
NUM_ROLLOUT=${NUM_ROLLOUT:-150}
NUM_GPUS=${NUM_GPUS:-2}
TP_SIZE=${TP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-32768}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-8192}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-5}

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
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --dynamic-sampling-filter-path generate_with_webshop.check_episode_reward_nonzero_std
   --custom-reward-post-process-path generate_with_webshop.grpo_normalize_webshop_steps
   --custom-rollout-log-function-path generate_with_webshop.log_webshop_rollout
   --balance-data
)

EVAL_ARGS=(
   --eval-interval "${EVAL_INTERVAL}"
   --eval-prompt-data valid "${WEBSHOP_TASK_DIR}/valid.jsonl"
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
   # --wandb-group Qwen2.5-3B-Instruct
   # --wandb-key ${WANDB_KEY}
)

SWANLAB_ARGS=(
   --use-swanlab
   --swanlab-mode "${SWANLAB_MODE:-cloud}"
   --swanlab-project "${SWANLAB_PROJECT:-slime-webshop}"
   --swanlab-group "${SWANLAB_GROUP:-Qwen2.5-3B-Instruct}"
   --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-Qwen2.5-3B-Instruct-grpo}"
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
   --sglang-mem-fraction-static 0.6
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
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/root/shared/ray_temp}
mkdir -p "${RAY_TEMP_DIR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir "${RAY_TEMP_DIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WEBSHOP_SERVICE_URL\": \"${WEBSHOP_SERVICE_URL}\",
    \"WEBSHOP_HISTORY_LENGTH\": \"${WEBSHOP_HISTORY_LENGTH}\",
    \"WEBSHOP_REWARD_MODE\": \"${WEBSHOP_REWARD_MODE}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
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
