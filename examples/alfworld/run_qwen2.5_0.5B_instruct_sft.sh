#!/bin/bash

# Docker-friendly ALFWorld SFT launcher for Qwen2.5-0.5B-Instruct.
# It trains the 0.5B student on messages JSONL built from successful 3B teacher
# trajectories, and can optionally run ALFWorld valid_seen/valid_unseen eval.

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
# Colocated eval enables SGLang's memory saver path; backend:native avoids the
# torch_memory_saver / expandable_segments incompatibility on affected stacks.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-backend:native}

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
SLIME_CKPT=${SLIME_CKPT:-/root/Qwen2.5-0.5B-Instruct_alfworld_sft_slime}
SFT_DATA=${SFT_DATA:-/root/slime-alfworld-teacher-sft/alfworld_teacher_sft.jsonl}
ALFWORLD_TASK_DIR=${ALFWORLD_TASK_DIR:-/root/slime-alfworld}
export ALFWORLD_DATA=${ALFWORLD_DATA:-/root/.cache/alfworld}
export ALFWORLD_CONFIG_PATH=${ALFWORLD_CONFIG_PATH:-${SCRIPT_DIR}/configs/config_tw.yaml}
export ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-50}
export ALFWORLD_HISTORY_LENGTH=${ALFWORLD_HISTORY_LENGTH:-4}
export ALFWORLD_STEP_MAX_TOKENS=${ALFWORLD_STEP_MAX_TOKENS:-512}
export ALFWORLD_INVALID_ACTION_PENALTY=${ALFWORLD_INVALID_ACTION_PENALTY:-0.01}
export ALFWORLD_ENV_WORKER_CPUS=${ALFWORLD_ENV_WORKER_CPUS:-0.1}
export ALFWORLD_ENV_WORKER_MAX_EPISODES=${ALFWORLD_ENV_WORKER_MAX_EPISODES:-1}

NUM_EPOCH=${NUM_EPOCH:-3}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}
NUM_GPUS=${NUM_GPUS:-4}
TP_SIZE=${TP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-32768}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-1024}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}
LR=${LR:-1e-5}
MIN_LR=${MIN_LR:-1e-6}
LR_WARMUP_FRACTION=${LR_WARMUP_FRACTION:-0.1}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}

# USE_EVAL=1 scores the SFT checkpoint on ALFWorld valid_seen/valid_unseen with
# the same batched rollout path as 3B GRPO. Set USE_EVAL=0 for training-only SFT.
USE_EVAL=${USE_EVAL:-1}

require_path() {
   local path="$1"
   local desc="$2"
   if [[ ! -e "${path}" ]]; then
      echo "ERROR: missing ${desc}: ${path}" >&2
      exit 2
   fi
}

require_path "${MODEL_ROOT}" "HF model root"
require_path "${MCORE_CKPT}" "Megatron torch_dist checkpoint"
require_path "${SFT_DATA}" "ALFWorld SFT messages data"
require_path "${ALFWORLD_CONFIG_PATH}" "ALFWorld config"
if [[ "${USE_EVAL}" == "1" ]]; then
   require_path "${ALFWORLD_TASK_DIR}/valid_seen_games.jsonl" "ALFWorld valid_seen game index"
   require_path "${ALFWORLD_TASK_DIR}/valid_unseen_games.jsonl" "ALFWorld valid_unseen game index"
fi
mkdir -p "${SLIME_CKPT}"

EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}
export ALFWORLD_EVAL_BATCH_SIZE=${ALFWORLD_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE}}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.7}

CKPT_ARGS=(
   --hf-checkpoint "${MODEL_ROOT}/"
   --ref-load "${MCORE_CKPT}/"
   --load "${SLIME_CKPT}/"
   --save "${SLIME_CKPT}/"
   --save-interval "${SAVE_INTERVAL}"
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${SFT_DATA}"
   --input-key messages
   --metadata-key metadata
   --loss-mask-type qwen
   --rollout-shuffle
   --num-epoch "${NUM_EPOCH}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --rollout-max-response-len "${ALFWORLD_STEP_MAX_TOKENS}"
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
)

EVAL_ARGS=()
SGLANG_ARGS=()
COLOCATE_ARGS=()
if [[ "${USE_EVAL}" == "1" ]]; then
   EVAL_ARGS=(
      --eval-function-path batched_rollout.generate_rollout
      --eval-interval "${EVAL_INTERVAL}"
      --eval-prompt-data valid_seen "${ALFWORLD_TASK_DIR}/valid_seen_games.jsonl" valid_unseen "${ALFWORLD_TASK_DIR}/valid_unseen_games.jsonl"
      --n-samples-per-eval-prompt 1
      --eval-max-response-len "${ALFWORLD_STEP_MAX_TOKENS}"
      --eval-top-k 1
   )
   SGLANG_ARGS=(
      --rollout-num-gpus "${NUM_GPUS}"
      --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
      --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
   )
   COLOCATE_ARGS=(--colocate --num-gpus-per-node "${NUM_GPUS}")
else
   SFT_ARGS+=(--debug-train-only)
fi

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
   --lr "${LR}"
   --lr-decay-style cosine
   --min-lr "${MIN_LR}"
   --lr-warmup-fraction "${LR_WARMUP_FRACTION}"
   --weight-decay "${WEIGHT_DECAY}"
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-alfworld
   # --wandb-group qwen2.5-0.5B-instruct-sft
   # --wandb-key ${WANDB_KEY}
)

SWANLAB_ARGS=()
if [[ "${USE_SWANLAB:-1}" == "1" ]]; then
   SWANLAB_ARGS=(
      --use-swanlab
      --swanlab-mode "${SWANLAB_MODE:-cloud}"
      --swanlab-project "${SWANLAB_PROJECT:-slime-alfworld}"
      --swanlab-group "${SWANLAB_GROUP:-qwen2.5-0.5B-instruct-sft}"
      --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-qwen2.5-0.5B-instruct-alfworld-sft}"
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

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
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
    \"PYTORCH_CUDA_ALLOC_CONF\": \"${PYTORCH_CUDA_ALLOC_CONF}\",
    \"ALFWORLD_DATA\": \"${ALFWORLD_DATA}\",
    \"ALFWORLD_CONFIG_PATH\": \"${ALFWORLD_CONFIG_PATH}\",
    \"ALFWORLD_MAX_STEPS\": \"${ALFWORLD_MAX_STEPS}\",
    \"ALFWORLD_HISTORY_LENGTH\": \"${ALFWORLD_HISTORY_LENGTH}\",
    \"ALFWORLD_STEP_MAX_TOKENS\": \"${ALFWORLD_STEP_MAX_TOKENS}\",
    \"ALFWORLD_INVALID_ACTION_PENALTY\": \"${ALFWORLD_INVALID_ACTION_PENALTY}\",
    \"ALFWORLD_ENV_WORKER_CPUS\": \"${ALFWORLD_ENV_WORKER_CPUS}\",
    \"ALFWORLD_ENV_WORKER_MAX_EPISODES\": \"${ALFWORLD_ENV_WORKER_MAX_EPISODES}\",
    \"ALFWORLD_EVAL_BATCH_SIZE\": \"${ALFWORLD_EVAL_BATCH_SIZE}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NUM_GPUS}" \
   "${COLOCATE_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${SWANLAB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}"
