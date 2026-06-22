#!/bin/bash

# Optional destructive cleanup for dedicated nodes only. It is disabled by
# default because broad pkill/ray stop can kill unrelated jobs on shared servers.
if [[ "${SCIENCEWORLD_FORCE_CLEANUP:-0}" == "1" ]]; then
   echo "SCIENCEWORLD_FORCE_CLEANUP=1: stopping local Ray/SGLang/Python processes before launch."
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
SLIME_CKPT=${SLIME_CKPT:-/root/Qwen2.5-3B-Instruct_scienceworld_grpo_slime}
SCIENCEWORLD_TASK_DIR=${SCIENCEWORLD_TASK_DIR:-/root/slime-scienceworld}
export SCIENCEWORLD_SIMPLIFICATION=${SCIENCEWORLD_SIMPLIFICATION:-easy}
export SCIENCEWORLD_JAR_PATH=${SCIENCEWORLD_JAR_PATH:-}
export SCIENCEWORLD_MAX_STEPS=${SCIENCEWORLD_MAX_STEPS:-50}
export SCIENCEWORLD_ENV_STEP_LIMIT=${SCIENCEWORLD_ENV_STEP_LIMIT:-100}
export SCIENCEWORLD_HISTORY_LENGTH=${SCIENCEWORLD_HISTORY_LENGTH:-4}
export SCIENCEWORLD_STEP_MAX_TOKENS=${SCIENCEWORLD_STEP_MAX_TOKENS:-512}
export SCIENCEWORLD_INVALID_ACTION_PENALTY=${SCIENCEWORLD_INVALID_ACTION_PENALTY:-0.01}
export SCIENCEWORLD_MAX_PROMPT_CHARS=${SCIENCEWORLD_MAX_PROMPT_CHARS:-12000}
export SCIENCEWORLD_ENV_WORKER_CPUS=${SCIENCEWORLD_ENV_WORKER_CPUS:-0.25}
export SCIENCEWORLD_ENV_WORKER_MAX_EPISODES=${SCIENCEWORLD_ENV_WORKER_MAX_EPISODES:-1}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
NUM_GPUS=${NUM_GPUS:-4}
TP_SIZE=${TP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}
export SCIENCEWORLD_EVAL_BATCH_SIZE=${SCIENCEWORLD_EVAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/root/shared/ray_temp}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/"
   --ref-load "${MCORE_CKPT}/"
   --load "${SLIME_CKPT}/"
   --save "${SLIME_CKPT}/"
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data "${SCIENCEWORLD_TASK_DIR}/train_tasks.jsonl"
   --input-key index
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${SCIENCEWORLD_STEP_MAX_TOKENS}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --dynamic-sampling-filter-path generate_with_scienceworld.check_episode_reward_nonzero_std
   --custom-reward-post-process-path generate_with_scienceworld.grpo_normalize_scienceworld_steps
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data eval "${SCIENCEWORLD_TASK_DIR}/eval_tasks.jsonl" test "${SCIENCEWORLD_TASK_DIR}/test_tasks.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len "${SCIENCEWORLD_STEP_MAX_TOKENS}"
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
   # --wandb-project slime-scienceworld
   # --wandb-group qwen2.5-3B-instruct
   # --wandb-key ${WANDB_KEY}
)

SWANLAB_ARGS=(
   --use-swanlab
   --swanlab-mode "${SWANLAB_MODE:-cloud}"
   --swanlab-project "${SWANLAB_PROJECT:-slime-scienceworld}"
   --swanlab-group "${SWANLAB_GROUP:-qwen2.5-3B-instruct}"
   --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-qwen2.5-3B-instruct-scienceworld-grpo}"
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
   --rollout-function-path batched_rollout.generate_rollout
)

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
mkdir -p "${RAY_TEMP_DIR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir "${RAY_TEMP_DIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"SCIENCEWORLD_SIMPLIFICATION\": \"${SCIENCEWORLD_SIMPLIFICATION}\",
    \"SCIENCEWORLD_JAR_PATH\": \"${SCIENCEWORLD_JAR_PATH}\",
    \"SCIENCEWORLD_MAX_STEPS\": \"${SCIENCEWORLD_MAX_STEPS}\",
    \"SCIENCEWORLD_ENV_STEP_LIMIT\": \"${SCIENCEWORLD_ENV_STEP_LIMIT}\",
    \"SCIENCEWORLD_HISTORY_LENGTH\": \"${SCIENCEWORLD_HISTORY_LENGTH}\",
    \"SCIENCEWORLD_STEP_MAX_TOKENS\": \"${SCIENCEWORLD_STEP_MAX_TOKENS}\",
    \"SCIENCEWORLD_INVALID_ACTION_PENALTY\": \"${SCIENCEWORLD_INVALID_ACTION_PENALTY}\",
    \"SCIENCEWORLD_MAX_PROMPT_CHARS\": \"${SCIENCEWORLD_MAX_PROMPT_CHARS}\",
    \"SCIENCEWORLD_ENV_WORKER_CPUS\": \"${SCIENCEWORLD_ENV_WORKER_CPUS}\",
    \"SCIENCEWORLD_ENV_WORKER_MAX_EPISODES\": \"${SCIENCEWORLD_ENV_WORKER_MAX_EPISODES}\",
    \"SCIENCEWORLD_EVAL_BATCH_SIZE\": \"${SCIENCEWORLD_EVAL_BATCH_SIZE}\"
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
