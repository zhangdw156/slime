#!/bin/bash

# Docker-friendly ALFWorld native-OPD launcher for Qwen2.5-0.5B-Instruct.
# The 0.5B student performs online ALFWorld rollouts. A trained 3B SGLang
# teacher endpoint scores the student's generated action tokens through slime's
# framework-native OPD helper; ALFWorld environment rewards are kept for metrics
# while processed training rewards are zeroed so OPD is the policy signal.

# Optional destructive cleanup for dedicated containers only. It is disabled by
# default because broad pkill/ray stop can kill unrelated jobs on shared servers.
if [[ "${ALFWORLD_FORCE_CLEANUP:-0}" == "1" ]]; then
   echo "ALFWORLD_FORCE_CLEANUP=1: stopping local Ray/SGLang/Python processes before launch."
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
source "${SCRIPT_DIR}/../../scripts/models/qwen2.5-0.5B.sh"

MODEL_ROOT=${MODEL_ROOT:-/root/Qwen2.5-0.5B-Instruct}
MCORE_CKPT=${MCORE_CKPT:-/root/Qwen2.5-0.5B-Instruct_torch_dist}
SLIME_CKPT=${SLIME_CKPT:-/root/Qwen2.5-0.5B-Instruct_alfworld_native_opd_from_3B_slime}
ALFWORLD_TASK_DIR=${ALFWORLD_TASK_DIR:-/root/slime-alfworld}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/root/.cache/alfworld}
export ALFWORLD_CONFIG_PATH=${ALFWORLD_CONFIG_PATH:-${SCRIPT_DIR}/configs/config_tw.yaml}
export ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-50}
export ALFWORLD_HISTORY_LENGTH=${ALFWORLD_HISTORY_LENGTH:-4}
export ALFWORLD_STEP_MAX_TOKENS=${ALFWORLD_STEP_MAX_TOKENS:-512}
export ALFWORLD_INVALID_ACTION_PENALTY=${ALFWORLD_INVALID_ACTION_PENALTY:-0.01}
export ALFWORLD_ENV_WORKER_CPUS=${ALFWORLD_ENV_WORKER_CPUS:-0.1}
export ALFWORLD_ENV_WORKER_MAX_EPISODES=${ALFWORLD_ENV_WORKER_MAX_EPISODES:-1}
# Route ALFWorld's custom rollout OPD annotation through slime's native
# on_policy_distillation.reward_func instead of the ALFWorld privileged-skill OPSD prompt.
export ALFWORLD_OPD_USE_NATIVE=1
export ALFWORLD_NATIVE_OPD_TEACHER_CONCURRENCY=${ALFWORLD_NATIVE_OPD_TEACHER_CONCURRENCY:-64}

# Point this at the trained 3B teacher SGLang /generate endpoint. The script
# does not launch or kill the teacher by default, so it is safe on shared nodes.
TEACHER_URL=${TEACHER_URL:-${ALFWORLD_OPD_TEACHER_URL:-http://127.0.0.1:30000/generate}}
export ALFWORLD_OPD_TEACHER_URL="${TEACHER_URL}"

require_path() {
   local path="$1"
   local desc="$2"
   if [[ ! -e "${path}" ]]; then
      echo "ERROR: missing ${desc}: ${path}" >&2
      exit 2
   fi
}

require_path "${MODEL_ROOT}" "0.5B HF model root"
require_path "${MCORE_CKPT}" "0.5B Megatron torch_dist checkpoint"
require_path "${ALFWORLD_CONFIG_PATH}" "ALFWorld config"
require_path "${ALFWORLD_TASK_DIR}/train_games.jsonl" "ALFWorld train game index"
require_path "${ALFWORLD_TASK_DIR}/valid_seen_games.jsonl" "ALFWorld valid_seen game index"
require_path "${ALFWORLD_TASK_DIR}/valid_unseen_games.jsonl" "ALFWorld valid_unseen game index"
mkdir -p "${SLIME_CKPT}"

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
NUM_GPUS=${NUM_GPUS:-4}
TP_SIZE=${TP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-32768}
OPD_KL_COEF=${OPD_KL_COEF:-1.0}
LR=${LR:-1e-6}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/"
   --ref-load "${MCORE_CKPT}/"
   --load "${SLIME_CKPT}/"
   --save "${SLIME_CKPT}/"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${ALFWORLD_TASK_DIR}/train_games.jsonl"
   --input-key index
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ALFWORLD_STEP_MAX_TOKENS}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --custom-reward-post-process-path generate_with_alfworld.zero_alfworld_rewards_for_opd
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data valid_seen "${ALFWORLD_TASK_DIR}/valid_seen_games.jsonl" valid_unseen "${ALFWORLD_TASK_DIR}/valid_unseen_games.jsonl"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len "${ALFWORLD_STEP_MAX_TOKENS}"
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

OPD_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-opd
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF}"
   --rm-url "${TEACHER_URL}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr "${LR}"
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-alfworld
   # --wandb-group qwen2.5-0.5B-instruct-native-opd
   # --wandb-key ${WANDB_KEY}
)

SWANLAB_ARGS=()
if [[ "${USE_SWANLAB:-1}" == "1" ]]; then
   SWANLAB_ARGS=(
      --use-swanlab
      --swanlab-mode "${SWANLAB_MODE:-cloud}"
      --swanlab-project "${SWANLAB_PROJECT:-slime-alfworld}"
      --swanlab-group "${SWANLAB_GROUP:-qwen2.5-0.5B-instruct-native-opd}"
      --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-qwen2.5-0.5B-instruct-alfworld-native-opd-from-3B}"
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
export no_proxy="127.0.0.1,${MASTER_ADDR}"
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/root/shared/ray_temp}
mkdir -p "${RAY_TEMP_DIR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir "${RAY_TEMP_DIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"ALFWORLD_DATA\": \"${ALFWORLD_DATA}\",
    \"ALFWORLD_CONFIG_PATH\": \"${ALFWORLD_CONFIG_PATH}\",
    \"ALFWORLD_MAX_STEPS\": \"${ALFWORLD_MAX_STEPS}\",
    \"ALFWORLD_HISTORY_LENGTH\": \"${ALFWORLD_HISTORY_LENGTH}\",
    \"ALFWORLD_STEP_MAX_TOKENS\": \"${ALFWORLD_STEP_MAX_TOKENS}\",
    \"ALFWORLD_INVALID_ACTION_PENALTY\": \"${ALFWORLD_INVALID_ACTION_PENALTY}\",
    \"ALFWORLD_ENV_WORKER_CPUS\": \"${ALFWORLD_ENV_WORKER_CPUS}\",
    \"ALFWORLD_ENV_WORKER_MAX_EPISODES\": \"${ALFWORLD_ENV_WORKER_MAX_EPISODES}\",
    \"ALFWORLD_OPD_USE_NATIVE\": \"${ALFWORLD_OPD_USE_NATIVE}\",
    \"ALFWORLD_NATIVE_OPD_TEACHER_CONCURRENCY\": \"${ALFWORLD_NATIVE_OPD_TEACHER_CONCURRENCY}\",
    \"ALFWORLD_OPD_TEACHER_URL\": \"${ALFWORLD_OPD_TEACHER_URL}\"
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
   "${OPD_ARGS[@]}" \
   "${DISTRIBUTED_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SWANLAB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
