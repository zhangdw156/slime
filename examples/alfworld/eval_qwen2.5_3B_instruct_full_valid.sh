#!/bin/bash

# Eval-only launcher for full ALFWorld valid_seen/valid_unseen evaluation.
# It regenerates separate full-valid JSONL indices from ALFWORLD_DATA, then
# starts slime with --num-rollout 0 so no training rollout/update is run.

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
SLIME_CKPT=${SLIME_CKPT:-/root/Qwen2.5-3B-Instruct_alfworld_grpo_slime}
# Optional. Eval-only does not need a reference model unless you explicitly add
# KL/OPD-related args, but this hook keeps parity with local checkpoint layouts.
MCORE_CKPT=${MCORE_CKPT:-}
CKPT_STEP=${CKPT_STEP:-}

export ALFWORLD_DATA=${ALFWORLD_DATA:-/root/.cache/alfworld}
export ALFWORLD_CONFIG_PATH=${ALFWORLD_CONFIG_PATH:-${SCRIPT_DIR}/configs/config_tw.yaml}
export ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-50}
export ALFWORLD_HISTORY_LENGTH=${ALFWORLD_HISTORY_LENGTH:-4}
export ALFWORLD_STEP_MAX_TOKENS=${ALFWORLD_STEP_MAX_TOKENS:-512}
export ALFWORLD_INVALID_ACTION_PENALTY=${ALFWORLD_INVALID_ACTION_PENALTY:-0.01}
export ALFWORLD_ENV_WORKER_CPUS=${ALFWORLD_ENV_WORKER_CPUS:-0.1}
export ALFWORLD_ENV_WORKER_MAX_EPISODES=${ALFWORLD_ENV_WORKER_MAX_EPISODES:-1}
export ALFWORLD_EVAL_BATCH_SIZE=${ALFWORLD_EVAL_BATCH_SIZE:-16}

ALFWORLD_FULL_EVAL_TASK_DIR=${ALFWORLD_FULL_EVAL_TASK_DIR:-/root/slime-alfworld-full-eval}
FULL_VALID_SEEN_DATA="${ALFWORLD_FULL_EVAL_TASK_DIR}/valid_seen_full_games.jsonl"
FULL_VALID_UNSEEN_DATA="${ALFWORLD_FULL_EVAL_TASK_DIR}/valid_unseen_full_games.jsonl"

if [[ ! -d "${MODEL_ROOT}" ]]; then
   echo "MODEL_ROOT does not exist: ${MODEL_ROOT}" >&2
   exit 1
fi
if [[ ! -d "${SLIME_CKPT}" ]]; then
   echo "SLIME_CKPT does not exist: ${SLIME_CKPT}" >&2
   exit 1
fi
if [[ -n "${MCORE_CKPT}" && ! -d "${MCORE_CKPT}" ]]; then
   echo "MCORE_CKPT is set but does not exist: ${MCORE_CKPT}" >&2
   exit 1
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

mkdir -p "${ALFWORLD_FULL_EVAL_TASK_DIR}"
ALFWORLD_SCRIPT_DIR="${SCRIPT_DIR}" \
ALFWORLD_FULL_EVAL_TASK_DIR="${ALFWORLD_FULL_EVAL_TASK_DIR}" \
python3 - <<'PY'
import os
import sys
from pathlib import Path

script_dir = Path(os.environ["ALFWORLD_SCRIPT_DIR"])
sys.path.insert(0, str(script_dir))

from prepare_alfworld_data import DEFAULT_TASK_TYPES, iter_games, write_split  # noqa: E402

alfworld_data = Path(os.environ["ALFWORLD_DATA"]).expanduser()
output_dir = Path(os.environ["ALFWORLD_FULL_EVAL_TASK_DIR"]).expanduser()
task_types = set(DEFAULT_TASK_TYPES)
outputs = {
    "valid_seen": output_dir / "valid_seen_full_games.jsonl",
    "valid_unseen": output_dir / "valid_unseen_full_games.jsonl",
}
for split, output_path in outputs.items():
    rows = list(iter_games(alfworld_data, split, task_types))
    if not rows:
        raise RuntimeError(f"No ALFWorld games found for {split} under {alfworld_data}")
    write_split(rows, output_path)
PY

ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
NUM_GPUS=${NUM_GPUS:-4}
TP_SIZE=${TP_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-12288}
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
   --disable-rollout-global-dataset
   --num-rollout 0
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-response-len "${ALFWORLD_STEP_MAX_TOKENS}"
   --rollout-temperature 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

EVAL_ARGS=(
   --eval-interval 1
   --eval-prompt-data valid_seen_full "${FULL_VALID_SEEN_DATA}" valid_unseen_full "${FULL_VALID_UNSEEN_DATA}"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len "${ALFWORLD_STEP_MAX_TOKENS}"
   --eval-top-k 1
)
if [[ -n "${EVAL_MAX_PROMPT_LEN:-}" ]]; then
   EVAL_ARGS+=(--eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}")
fi
if [[ -n "${EVAL_MAX_CONTEXT_LEN:-}" ]]; then
   EVAL_ARGS+=(--eval-max-context-len "${EVAL_MAX_CONTEXT_LEN}")
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
      --wandb-project "${WANDB_PROJECT:-slime-alfworld}"
      --wandb-group "${WANDB_GROUP:-qwen2.5-3B-instruct-alfworld-eval-${EVAL_TAG}}"
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
      --swanlab-project "${SWANLAB_PROJECT:-slime-alfworld}"
      --swanlab-group "${SWANLAB_GROUP:-qwen2.5-3B-instruct-eval}"
      --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME:-qwen2.5-3B-instruct-alfworld-eval-${EVAL_TAG}}"
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
      --tb-project-name "${TB_PROJECT_NAME:-slime-alfworld}"
      --tb-experiment-name "${TB_EXPERIMENT_NAME:-qwen2.5-3B-instruct-alfworld-eval-${EVAL_TAG}}"
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.7}"
)
if [[ -n "${SGLANG_SERVER_CONCURRENCY:-}" ]]; then
   SGLANG_ARGS+=(--sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}")
fi

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
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/root/shared/ray_temp}
mkdir -p "${RAY_TEMP_DIR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" --temp-dir "${RAY_TEMP_DIR}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:${SCRIPT_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
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
