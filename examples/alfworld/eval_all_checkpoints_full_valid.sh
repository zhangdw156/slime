#!/usr/bin/env bash
# Evaluate every saved ALFWorld checkpoint on full valid_seen/full valid_unseen,
# and log all checkpoint metrics into one SwanLab experiment.
#
# Default behavior is eval-only.  If you want this script to run the SFT launcher
# first and then evaluate all saved checkpoints, set RUN_TRAINING=1.  If training
# is already running, set WAIT_PID=<pid> to wait for it before scanning ckpts.

set -euo pipefail

if [[ "${DEBUG:-0}" == "1" ]]; then
  set -x
fi

export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${SCRIPT_DIR}/run_qwen2.5_3B_instruct_sft_full_eval.sh}
CUSTOM_LOGGER_PATH=${CUSTOM_LOGGER_PATH:-${SCRIPT_DIR}/checkpoint_eval_logger.py}
CUSTOM_LOGGER_FUNC=${CUSTOM_LOGGER_FUNC:-checkpoint_eval_logger.log_eval_at_checkpoint_step}

# -------- Paths: defaults target the 3B SFT sweep; override for other model families. --------
SLIME_ROOT=${SLIME_ROOT:-/data/zhangdw12/work/slime}
MEGATRON_ROOT=${MEGATRON_ROOT:-/data/zhangdw12/work/Megatron-LM}
MODEL_ARGS_SCRIPT=${MODEL_ARGS_SCRIPT:-qwen2.5-3B.sh}
MODEL_ROOT=${MODEL_ROOT:-/data/zhangdw12/models/Qwen2.5-3B-Instruct}
MCORE_CKPT=${MCORE_CKPT:-/data/zhangdw12/models/Qwen2.5-3B-Instruct_torch_dist}
SLIME_CKPT=${SLIME_CKPT:-/data/zhangdw12/models/Qwen2.5-3B-Instruct_alfworld_sft_slime}
ALFWORLD_DATA=${ALFWORLD_DATA:-/data/zhangdw12/datasets/alfworld}
ALFWORLD_EXAMPLE_DIR=${ALFWORLD_EXAMPLE_DIR:-${SLIME_ROOT}/examples/alfworld}
ALFWORLD_CONFIG_PATH=${ALFWORLD_CONFIG_PATH:-${ALFWORLD_EXAMPLE_DIR}/configs/config_tw.yaml}
ALFWORLD_FULL_EVAL_TASK_DIR=${ALFWORLD_FULL_EVAL_TASK_DIR:-/data/zhangdw12/datasets/slime-alfworld-full-eval}
FULL_VALID_SEEN_DATA=${ALFWORLD_FULL_EVAL_TASK_DIR}/valid_seen_full_games.jsonl
FULL_VALID_UNSEEN_DATA=${ALFWORLD_FULL_EVAL_TASK_DIR}/valid_unseen_full_games.jsonl

# -------- Optional sequencing. --------
RUN_TRAINING=${RUN_TRAINING:-0}
WAIT_PID=${WAIT_PID:-}

# -------- Runtime defaults. --------
export ALFWORLD_DATA
export ALFWORLD_CONFIG_PATH
export ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-50}
export ALFWORLD_HISTORY_LENGTH=${ALFWORLD_HISTORY_LENGTH:-4}
export ALFWORLD_STEP_MAX_TOKENS=${ALFWORLD_STEP_MAX_TOKENS:-512}
export ALFWORLD_INVALID_ACTION_PENALTY=${ALFWORLD_INVALID_ACTION_PENALTY:-0.01}
export ALFWORLD_ENV_WORKER_CPUS=${ALFWORLD_ENV_WORKER_CPUS:-0.1}
export ALFWORLD_ENV_WORKER_MAX_EPISODES=${ALFWORLD_ENV_WORKER_MAX_EPISODES:-1}
export ALFWORLD_EVAL_BATCH_SIZE=${ALFWORLD_EVAL_BATCH_SIZE:-16}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}
TP_SIZE=${TP_SIZE:-1}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.7}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-12288}

# -------- Ray isolation: only pin Ray head/GCS; dashboard is disabled. --------
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
START_RAY=${START_RAY:-1}
RAY_PORT_BASE=${RAY_PORT_BASE:-auto}
RAY_CORE_PORT_SEARCH_START=${RAY_CORE_PORT_SEARCH_START:-29379}
RAY_CORE_PORT_SEARCH_END=${RAY_CORE_PORT_SEARCH_END:-39999}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
RAY_TEMP_ROOT=${RAY_TEMP_ROOT:-/data/zhangdw12/r}
RAY_TMP_PREFIX=${RAY_TMP_PREFIX:-ra_sft_ckpt}
JOB_TMP_PREFIX=${JOB_TMP_PREFIX:-alfworld_sft_ckpt_eval}
RAY_CLEANUP_ON_EXIT=${RAY_CLEANUP_ON_EXIT:-1}
RAY_CLEAN_STALE_ON_START=${RAY_CLEAN_STALE_ON_START:-1}
ACTIVE_RAY_TMPDIR=""

# -------- One SwanLab experiment for the whole checkpoint sweep. --------
# Derive tracker defaults from the selected model-args script so overriding
# MODEL_ARGS_SCRIPT for 0.5B does not accidentally keep 3B tracker names.
case "${MODEL_ARGS_SCRIPT}" in
  qwen2.5-0.5B.sh)
    DEFAULT_SWANLAB_GROUP="qwen2.5-0.5B-instruct-sft-all-ckpts-full-eval"
    DEFAULT_SWANLAB_EXPERIMENT_NAME="qwen2.5-0.5B-instruct-sft-all-checkpoints-full-valid"
    DEFAULT_SWEEP_ID_PREFIX="sft0p5b-allckpt"
    ;;
  qwen2.5-3B.sh)
    DEFAULT_SWANLAB_GROUP="qwen2.5-3B-instruct-sft-all-ckpts-full-eval"
    DEFAULT_SWANLAB_EXPERIMENT_NAME="qwen2.5-3B-instruct-sft-all-checkpoints-full-valid"
    DEFAULT_SWEEP_ID_PREFIX="sft3b-allckpt"
    ;;
  *)
    model_label=$(basename "${MODEL_ARGS_SCRIPT}" .sh)
    model_label=${model_label//[^A-Za-z0-9._-]/-}
    DEFAULT_SWANLAB_GROUP="${model_label}-sft-all-ckpts-full-eval"
    DEFAULT_SWANLAB_EXPERIMENT_NAME="${model_label}-sft-all-checkpoints-full-valid"
    DEFAULT_SWEEP_ID_PREFIX="${model_label}-allckpt"
    ;;
esac
EVAL_TRACKING_ROOT=${EVAL_TRACKING_ROOT:-${SLIME_CKPT}/all_ckpt_full_eval_tracking}
SWANLAB_PROJECT=${SWANLAB_PROJECT:-slime-alfworld}
SWANLAB_GROUP=${SWANLAB_GROUP:-${DEFAULT_SWANLAB_GROUP}}
SWANLAB_EXPERIMENT_NAME=${SWANLAB_EXPERIMENT_NAME:-${DEFAULT_SWANLAB_EXPERIMENT_NAME}}
SWEEP_ID_PREFIX=${SWEEP_ID_PREFIX:-${DEFAULT_SWEEP_ID_PREFIX}}

# SWEEP_ID isolates local logs/state/done markers for this all-checkpoint eval.
# Default is a fresh id to avoid silently mixing a new evaluation with old done
# markers. To resume a failed sweep into the same SwanLab run, rerun with the
# printed SWEEP_ID (or SWANLAB_RUN_ID) and leave SKIP_EXISTING=1.
if [[ -z "${SWEEP_ID:-}" ]]; then
  SWEEP_ID=${TRAIN_RUN_ID:-${SWANLAB_RUN_ID:-}}
fi
if [[ -z "${SWEEP_ID:-}" ]]; then
  SWEEP_ID="${SWEEP_ID_PREFIX}-$(date +%y%m%d-%H%M%S)"
fi
case "${SWEEP_ID}" in
  *"/"*|*".."*|"")
    echo "ERROR: SWEEP_ID must be a non-empty path-safe id without '/' or '..': ${SWEEP_ID}" >&2
    exit 2
    ;;
esac

EVAL_LOG_DIR=${EVAL_LOG_DIR:-${EVAL_TRACKING_ROOT}/logs/${SWEEP_ID}}
SWEEP_STATE_DIR=${SWEEP_STATE_DIR:-${EVAL_TRACKING_ROOT}/state/${SWEEP_ID}}

USE_SWANLAB=${USE_SWANLAB:-1}
USE_TENSORBOARD=${USE_TENSORBOARD:-1}
TB_PROJECT_NAME=${TB_PROJECT_NAME:-${SWANLAB_PROJECT}}
TB_EXPERIMENT_NAME=${TB_EXPERIMENT_NAME:-${SWANLAB_EXPERIMENT_NAME}}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${EVAL_TRACKING_ROOT}/tensorboard/${TB_PROJECT_NAME}/${TB_EXPERIMENT_NAME}/${SWEEP_ID}}

# CKPT_STEPS=all scans iter_* directories.  Or pass comma-separated steps, e.g. CKPT_STEPS=100,200,300.
CKPT_STEPS=${CKPT_STEPS:-all}
SKIP_EXISTING=${SKIP_EXISTING:-1}

require_path() {
  local path="$1"
  local desc="$2"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing ${desc}: ${path}" >&2
    exit 2
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: missing required command: ${cmd}" >&2
    exit 2
  fi
}

init_sweep_state() {
  mkdir -p "${EVAL_LOG_DIR}" "${SWEEP_STATE_DIR}"

  local run_id_file="${SWEEP_STATE_DIR}/swanlab_run_id.txt"
  if [[ -z "${SWANLAB_RUN_ID:-}" ]]; then
    if [[ -s "${run_id_file}" ]]; then
      SWANLAB_RUN_ID=$(tr -d '[:space:]' < "${run_id_file}")
    else
      SWANLAB_RUN_ID="${SWEEP_ID}"
      printf '%s\n' "${SWANLAB_RUN_ID}" > "${run_id_file}"
    fi
  elif [[ -s "${run_id_file}" ]]; then
    local saved_run_id
    saved_run_id=$(tr -d '[:space:]' < "${run_id_file}")
    if [[ "${saved_run_id}" != "${SWANLAB_RUN_ID}" && "${ALLOW_SWANLAB_RUN_ID_MISMATCH:-0}" != "1" ]]; then
      echo "ERROR: SWEEP_STATE_DIR already belongs to SwanLab run_id=${saved_run_id}, but SWANLAB_RUN_ID=${SWANLAB_RUN_ID}." >&2
      echo "Use a new SWEEP_ID/SWEEP_STATE_DIR for a new experiment, or set ALLOW_SWANLAB_RUN_ID_MISMATCH=1 intentionally." >&2
      exit 2
    fi
  else
    printf '%s\n' "${SWANLAB_RUN_ID}" > "${run_id_file}"
  fi
  export SWANLAB_RUN_ID

  {
    printf 'sweep_id=%s\n' "${SWEEP_ID}"
    printf 'swanlab_run_id=%s\n' "${SWANLAB_RUN_ID}"
    printf 'swanlab_project=%s\n' "${SWANLAB_PROJECT}"
    printf 'swanlab_group=%s\n' "${SWANLAB_GROUP}"
    printf 'swanlab_experiment_name=%s\n' "${SWANLAB_EXPERIMENT_NAME}"
    printf 'model_args_script=%s\n' "${MODEL_ARGS_SCRIPT}"
    printf 'model_root=%s\n' "${MODEL_ROOT}"
    printf 'mcore_ckpt=%s\n' "${MCORE_CKPT}"
    printf 'slime_ckpt=%s\n' "${SLIME_CKPT}"
    date -Iseconds | sed 's/^/created_or_resumed_at=/'
  } > "${SWEEP_STATE_DIR}/sweep_metadata.env"
}

check_ray_temp_dir_short() {
  local path="$1"
  python3 - "${path}" <<'PY_RAY_TEMP_LEN'
import sys
path = sys.argv[1].rstrip("/")
if len(path.encode()) > 40:
    raise SystemExit(
        f"ERROR: Ray temp dir is too long for AF_UNIX sockets ({len(path.encode())} bytes): {path}\n"
        "Use a shorter RAY_TEMP_ROOT, e.g. /data/zhangdw12/r."
    )
PY_RAY_TEMP_LEN
}

check_port_free() {
  local port="$1"
  local name="$2"
  python3 - "${port}" "${name}" <<'PY_PORT_FREE'
import socket
import sys
port = int(sys.argv[1])
name = sys.argv[2]
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("", port))
except OSError as exc:
    raise SystemExit(f"ERROR: {name} port {port} is already in use: {exc}")
finally:
    sock.close()
PY_PORT_FREE
}

find_free_ray_port() {
  local search_start="$1"
  local search_end="$2"
  python3 - "${search_start}" "${search_end}" <<'PY_FIND_RAY_PORT'
import socket
import sys
start = int(sys.argv[1])
end = int(sys.argv[2])
for port in range(start, end + 1):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("", port))
        print(port)
        raise SystemExit(0)
    except OSError:
        pass
    finally:
        sock.close()
raise SystemExit(f"ERROR: cannot find free Ray head port in {start}-{end}")
PY_FIND_RAY_PORT
}

cleanup_own_ray_processes() {
  local temp_dir="$1"
  [[ -n "${temp_dir}" ]] || return 0
  python3 - "${temp_dir}" <<'PY_CLEAN_RAY'
import os
import signal
import sys
import time
root = os.path.realpath(sys.argv[1])
if not root or root == "/":
    raise SystemExit("Refusing to clean unsafe Ray temp root")
self_pids = {os.getpid(), os.getppid()}
ray_tokens = ("ray", "gcs_server", "raylet", "dashboard")

def read_cmdline(pid):
    try:
        with open(os.path.join("/proc", str(pid), "cmdline"), "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        return ""

def is_script_owned_ray(pid):
    if pid in self_pids:
        return False
    cmdline = read_cmdline(pid)
    return bool(cmdline) and root in cmdline and any(token in cmdline for token in ray_tokens)

pids = set()
proc = "/proc"
if os.path.isdir(proc):
    for name in os.listdir(proc):
        if name.isdigit() and is_script_owned_ray(int(name)):
            pids.add(int(name))
for base, dirs, files in os.walk(root):
    rel = os.path.relpath(base, root)
    if rel != "." and rel.count(os.sep) >= 4:
        dirs[:] = []
    for filename in files:
        if not filename.endswith(".pid"):
            continue
        try:
            pid = int(open(os.path.join(base, filename), encoding="utf-8").read().strip().split()[0])
        except Exception:
            continue
        if is_script_owned_ray(pid):
            pids.add(pid)
if not pids:
    sys.exit(0)
print(f"Cleaning script-owned Ray processes under {root}: {sorted(pids)}", flush=True)
for pid in sorted(pids):
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        print(f"WARNING: no permission to terminate pid {pid}", file=sys.stderr, flush=True)
deadline = time.time() + 8
while time.time() < deadline:
    alive = []
    for pid in sorted(pids):
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except ProcessLookupError:
            pass
    if not alive:
        break
    time.sleep(0.2)
else:
    for pid in alive:
        if not is_script_owned_ray(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"WARNING: no permission to kill pid {pid}", file=sys.stderr, flush=True)
PY_CLEAN_RAY
}

cleanup_active_ray_on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${ACTIVE_RAY_TMPDIR:-}" && "${START_RAY}" == "1" && "${RAY_CLEANUP_ON_EXIT}" == "1" ]]; then
    cleanup_own_ray_processes "${ACTIVE_RAY_TMPDIR}" || true
  fi
  exit "${exit_code}"
}
trap cleanup_active_ray_on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

preflight() {
  require_command python3
  if [[ "${START_RAY}" == "1" ]]; then
    require_command ray
  fi
  if [[ "${RUN_TRAINING}" == "1" ]]; then
    if [[ ! -e "${TRAIN_SCRIPT}" ]]; then
      echo "ERROR: missing training script: ${TRAIN_SCRIPT}" >&2
      echo "Hint: launch through the Nutstore wrapper or set TRAIN_SCRIPT=/path/to/run_qwen2.5_3B_instruct_sft_full_eval.sh." >&2
      exit 2
    fi
  fi
  require_path "${CUSTOM_LOGGER_PATH}" "custom checkpoint eval logger"
  require_path "${SLIME_ROOT}/train.py" "slime train.py"
  require_path "${SLIME_ROOT}/scripts/models/${MODEL_ARGS_SCRIPT}" "model args script ${MODEL_ARGS_SCRIPT}"
  require_path "${MEGATRON_ROOT}" "Megatron-LM root"
  require_path "${MODEL_ROOT}" "HF model root"
  require_path "${MCORE_CKPT}/latest_checkpointed_iteration.txt" "Megatron torch_dist checkpoint marker"
  require_path "${ALFWORLD_EXAMPLE_DIR}/batched_rollout.py" "ALFWorld batched rollout"
  require_path "${ALFWORLD_EXAMPLE_DIR}/prepare_alfworld_data.py" "ALFWorld data preparer"
  require_path "${ALFWORLD_CONFIG_PATH}" "ALFWorld config"
  require_path "${ALFWORLD_DATA}/json_2.1.1/valid_seen" "ALFWorld valid_seen data"
  require_path "${ALFWORLD_DATA}/json_2.1.1/valid_unseen" "ALFWorld valid_unseen data"
}

maybe_run_or_wait_training() {
  if [[ "${RUN_TRAINING}" == "1" ]]; then
    echo "RUN_TRAINING=1: launching ${TRAIN_SCRIPT}; disabling its built-in latest-only full eval."
    RUN_FULL_EVAL_AFTER_SFT=0 bash "${TRAIN_SCRIPT}"
  elif [[ -n "${WAIT_PID}" ]]; then
    echo "Waiting for training PID ${WAIT_PID} to exit before checkpoint sweep..."
    while ps -p "${WAIT_PID}" >/dev/null 2>&1; do
      sleep 60
    done
    echo "PID ${WAIT_PID} exited; starting checkpoint sweep."
  fi
}

generate_full_eval_indices() {
  mkdir -p "${ALFWORLD_FULL_EVAL_TASK_DIR}"
  ALFWORLD_SCRIPT_DIR="${ALFWORLD_EXAMPLE_DIR}" \
  ALFWORLD_FULL_EVAL_TASK_DIR="${ALFWORLD_FULL_EVAL_TASK_DIR}" \
  python3 - <<'PY_FULL_EVAL_INDEX'
import os
import sys
from pathlib import Path
script_dir = Path(os.environ["ALFWORLD_SCRIPT_DIR"])
sys.path.insert(0, str(script_dir))
from prepare_alfworld_data import DEFAULT_TASK_TYPES, iter_games, write_split  # noqa: E402
alfworld_data = Path(os.environ["ALFWORLD_DATA"]).expanduser()
output_dir = Path(os.environ["ALFWORLD_FULL_EVAL_TASK_DIR"]).expanduser()
task_types = set(DEFAULT_TASK_TYPES)
for split, filename in {
    "valid_seen": "valid_seen_full_games.jsonl",
    "valid_unseen": "valid_unseen_full_games.jsonl",
}.items():
    rows = list(iter_games(alfworld_data, split, task_types))
    if not rows:
        raise RuntimeError(f"No ALFWorld games found for {split} under {alfworld_data}")
    output_path = output_dir / filename
    write_split(rows, output_path)
    print(f"Wrote {len(rows)} {split} games to {output_path}", flush=True)
PY_FULL_EVAL_INDEX
}

selected_ckpt_steps() {
  if [[ "${CKPT_STEPS}" != "all" ]]; then
    local item
    IFS=',' read -r -a parsed_steps <<< "${CKPT_STEPS}"
    for item in "${parsed_steps[@]}"; do
      item=${item//[[:space:]]/}
      [[ -n "${item}" ]] || continue
      if [[ ! "${item}" =~ ^[0-9]+$ ]]; then
        echo "ERROR: CKPT_STEPS must be 'all' or comma-separated non-negative integers; got: ${item}" >&2
        exit 2
      fi
      printf '%s\n' "${item}"
    done
    return 0
  fi
  python3 - "${SLIME_CKPT}" <<'PY_DISCOVER_CKPTS'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
steps = []
for path in root.glob("iter_*"):
    if not path.is_dir():
        continue
    m = re.fullmatch(r"iter_(\d+)", path.name)
    if m:
        steps.append(int(m.group(1)))
for step in sorted(set(steps)):
    print(step)
PY_DISCOVER_CKPTS
}

start_ray_for_step() {
  local idx="$1"
  local step="$2"

  if [[ "${START_RAY}" == "0" ]]; then
    if [[ -z "${RAY_ADDRESS:-}" ]]; then
      echo "ERROR: START_RAY=0 requires RAY_ADDRESS=<host:port> for the existing Ray cluster." >&2
      exit 2
    fi
    local job_tmp="${JOB_TMP_BASE:-/data/zhangdw12/tmp}/${JOB_TMP_PREFIX}_${step}_${SLURM_JOB_ID:-$$}"
    mkdir -p "${job_tmp}"
    export TMPDIR="${job_tmp}"
    export TEMP="${job_tmp}"
    export TMP="${job_tmp}"
    ACTIVE_RAY_TMPDIR=""
    echo "START_RAY=0: reusing existing Ray at ${RAY_ADDRESS} for checkpoint ${step}"
    return 0
  fi

  local eval_ray_port
  if [[ "${RAY_PORT_BASE}" == "auto" ]]; then
    eval_ray_port=$(find_free_ray_port "${RAY_CORE_PORT_SEARCH_START}" "${RAY_CORE_PORT_SEARCH_END}")
  else
    eval_ray_port=$((RAY_PORT_BASE + idx))
  fi

  local ray_tmpdir="${RAY_TEMP_ROOT%/}/${RAY_TMP_PREFIX}_${step}"
  local job_tmp="${JOB_TMP_BASE:-/data/zhangdw12/tmp}/${JOB_TMP_PREFIX}_${step}_${SLURM_JOB_ID:-$$}"
  check_ray_temp_dir_short "${ray_tmpdir}"
  mkdir -p "${ray_tmpdir}" "${job_tmp}"
  if [[ "${RAY_CLEAN_STALE_ON_START}" == "1" ]]; then
    cleanup_own_ray_processes "${ray_tmpdir}" || true
  fi

  export RAY_ADDRESS="${MASTER_ADDR}:${eval_ray_port}"
  export RAY_TMPDIR="${ray_tmpdir}"
  export TMPDIR="${job_tmp}"
  export TEMP="${job_tmp}"
  export TMP="${job_tmp}"
  ACTIVE_RAY_TMPDIR="${ray_tmpdir}"

  check_port_free "${eval_ray_port}" "Ray GCS"
  echo "Starting Ray for checkpoint ${step}: ${RAY_ADDRESS}, dashboard disabled, temp ${ray_tmpdir}"
  ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --port "${eval_ray_port}" \
    --include-dashboard=false \
    --num-cpus "${RAY_NUM_CPUS}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --temp-dir "${ray_tmpdir}"
}
stop_active_ray() {
  if [[ -n "${ACTIVE_RAY_TMPDIR:-}" && "${START_RAY}" == "1" && "${RAY_CLEANUP_ON_EXIT}" == "1" ]]; then
    cleanup_own_ray_processes "${ACTIVE_RAY_TMPDIR}" || true
  fi
  ACTIVE_RAY_TMPDIR=""
}

run_eval_step() {
  local step="$1"
  local ckpt_iter_dir
  printf -v ckpt_iter_dir 'iter_%07d' "${step}"
  require_path "${SLIME_CKPT}/${ckpt_iter_dir}" "SFT checkpoint ${ckpt_iter_dir}"
  if [[ "${REQUIRE_CKPT_NONEMPTY:-1}" == "1" ]]; then
    if [[ -z "$(find "${SLIME_CKPT}/${ckpt_iter_dir}" -mindepth 1 -print -quit)" ]]; then
      echo "ERROR: SFT checkpoint ${SLIME_CKPT}/${ckpt_iter_dir} exists but is empty; refusing to evaluate it." >&2
      exit 2
    fi
  fi
  require_path "${FULL_VALID_SEEN_DATA}" "full valid_seen index"
  require_path "${FULL_VALID_UNSEEN_DATA}" "full valid_unseen index"

  local done_marker="${SWEEP_STATE_DIR}/step_${step}.done"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${done_marker}" ]]; then
    local marker_run_id=""
    local marker_sweep_id=""
    marker_run_id=$(awk -F= '$1 == "swanlab_run_id" {print substr($0, index($0, "=") + 1)}' "${done_marker}" | tail -n 1)
    marker_sweep_id=$(awk -F= '$1 == "sweep_id" {print substr($0, index($0, "=") + 1)}' "${done_marker}" | tail -n 1)
    if [[ "${marker_run_id}" == "${SWANLAB_RUN_ID}" && "${marker_sweep_id}" == "${SWEEP_ID}" ]]; then
      echo "Skipping checkpoint ${step}: already done for SWEEP_ID=${SWEEP_ID}, run_id=${SWANLAB_RUN_ID}. Set SKIP_EXISTING=0 to rerun."
      return 0
    fi
    echo "Ignoring stale/incompatible done marker for checkpoint ${step}: ${done_marker}" >&2
  fi

  cd "${SLIME_ROOT}"
  unset MODEL_ARGS || true
  # shellcheck source=/dev/null
  source "${SLIME_ROOT}/scripts/models/${MODEL_ARGS_SCRIPT}"
  if [[ "${#MODEL_ARGS[@]}" -eq 0 ]]; then
    echo "ERROR: ${MODEL_ARGS_SCRIPT} did not define MODEL_ARGS" >&2
    exit 2
  fi
  export PYTHONPATH="${SCRIPT_DIR}:${MEGATRON_ROOT}:${SLIME_ROOT}:${ALFWORLD_EXAMPLE_DIR}:${PYTHONPATH:-}"
  export CUDA_DEVICE_MAX_CONNECTIONS=1
  export CHECKPOINT_EVAL_STEP="${step}"
  export ALFWORLD_EVAL_CKPT_STEP="${step}"
  mkdir -p "${TENSORBOARD_DIR}"

  local CKPT_ARGS=(
    --hf-checkpoint "${MODEL_ROOT}/"
    --ref-load "${MCORE_CKPT}/"
    --load "${SLIME_CKPT}/"
    --ckpt-step "${step}"
    --no-load-optim
    --no-load-rng
    --finetune
  )
  local ROLLOUT_ARGS=(
    --disable-rollout-global-dataset
    --num-rollout 0
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ALFWORLD_STEP_MAX_TOKENS}"
    --rollout-temperature 1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
  )
  local EVAL_ARGS=(
    --eval-interval 1
    --eval-prompt-data valid_seen_full "${FULL_VALID_SEEN_DATA}" valid_unseen_full "${FULL_VALID_UNSEEN_DATA}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${ALFWORLD_STEP_MAX_TOKENS}"
    --eval-top-k 1
    --custom-eval-rollout-log-function-path "${CUSTOM_LOGGER_FUNC}"
  )
  local PERF_ARGS=(
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
  local OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --lr-decay-iters "${LR_DECAY_ITERS:-1}"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
  )
  local SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  )
  if [[ -n "${SGLANG_SERVER_CONCURRENCY:-}" ]]; then
    SGLANG_ARGS+=(--sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}")
  fi
  local CUSTOM_ARGS=(
    --rollout-function-path batched_rollout.generate_rollout
    --eval-function-path batched_rollout.generate_rollout
  )
  local MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
  )
  local TRACKING_ARGS=()
  if [[ "${USE_SWANLAB}" == "1" ]]; then
    TRACKING_ARGS+=(
      --use-swanlab
      --swanlab-mode "${SWANLAB_MODE:-cloud}"
      --swanlab-project "${SWANLAB_PROJECT}"
      --swanlab-group "${SWANLAB_GROUP}"
      --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME}"
      --swanlab-run-id "${SWANLAB_RUN_ID}"
      --disable-swanlab-random-suffix
    )
    if [[ "${SWANLAB_MODE:-cloud}" == "cloud" && -n "${SWANLAB_API_KEY:-}" ]]; then
      TRACKING_ARGS+=(--swanlab-key "${SWANLAB_API_KEY}")
    fi
    if [[ -n "${SWANLAB_WORKSPACE:-}" ]]; then
      TRACKING_ARGS+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
    fi
    if [[ -n "${SWANLAB_DIR:-}" ]]; then
      TRACKING_ARGS+=(--swanlab-dir "${SWANLAB_DIR}")
    fi
  fi
  if [[ "${USE_TENSORBOARD}" == "1" ]]; then
    TRACKING_ARGS+=(
      --use-tensorboard
      --tb-project-name "${TB_PROJECT_NAME}"
      --tb-experiment-name "${TB_EXPERIMENT_NAME}"
    )
  fi

  local TRAIN_ARGS=(
    --actor-num-nodes 1
    --actor-num-gpus-per-node "${NUM_GPUS}"
    --rollout-num-gpus "${NUM_GPUS}"
    --colocate
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${TRACKING_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${EVAL_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${CUSTOM_ARGS[@]}"
  )

  local log_file="${EVAL_LOG_DIR}/eval_step_${step}.log"
  echo "========== Full-valid eval checkpoint ${step} ==========" | tee "${log_file}"
  echo "Model args script: ${MODEL_ARGS_SCRIPT}" | tee -a "${log_file}"
  echo "SwanLab: project=${SWANLAB_PROJECT}, experiment=${SWANLAB_EXPERIMENT_NAME}, run_id=${SWANLAB_RUN_ID}, x-axis step=${step}" | tee -a "${log_file}"

  set +e
  python3 - "${TRAIN_ARGS[@]}" <<'PY_DRIVER' 2>&1 | tee -a "${log_file}"
import faulthandler
import os
import runpy
import signal
import sys

import ray

faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
except (AttributeError, RuntimeError, ValueError):
    pass
stack_dump_after = int(os.environ.get("PYTHON_STACK_DUMP_AFTER", "120") or "0")
if stack_dump_after > 0:
    faulthandler.dump_traceback_later(stack_dump_after, repeat=True)

address = os.environ.get("RAY_ADDRESS")
if not address:
    raise RuntimeError("RAY_ADDRESS is not set")

env_keys = (
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "TMPDIR",
    "TEMP",
    "TMP",
    "RAY_TMPDIR",
    "PATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CONDA_PREFIX",
    "VIRTUAL_ENV",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "ALFWORLD_DATA",
    "ALFWORLD_CONFIG_PATH",
    "ALFWORLD_MAX_STEPS",
    "ALFWORLD_HISTORY_LENGTH",
    "ALFWORLD_STEP_MAX_TOKENS",
    "ALFWORLD_INVALID_ACTION_PENALTY",
    "ALFWORLD_ENV_WORKER_CPUS",
    "ALFWORLD_ENV_WORKER_MAX_EPISODES",
    "ALFWORLD_EVAL_BATCH_SIZE",
    "CHECKPOINT_EVAL_STEP",
    "ALFWORLD_EVAL_CKPT_STEP",
    "TENSORBOARD_DIR",
)
env_vars = {key: os.environ[key] for key in env_keys if os.environ.get(key)}
print(f"Connecting Ray driver to {address}", flush=True)
print(f"Ray runtime env vars: {env_vars}", flush=True)
ray.init(address=address, runtime_env={"env_vars": env_vars})
print("Ray driver connected; entering train.py for checkpoint full-valid eval", flush=True)
sys.argv = ["train.py", *sys.argv[1:]]
runpy.run_path("train.py", run_name="__main__")
PY_DRIVER
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "ERROR: checkpoint ${step} full-valid eval failed; see ${log_file}" >&2
    return "${status}"
  fi

  {
    printf 'step=%s\n' "${step}"
    printf 'sweep_id=%s\n' "${SWEEP_ID}"
    printf 'swanlab_run_id=%s\n' "${SWANLAB_RUN_ID}"
    printf 'swanlab_experiment_name=%s\n' "${SWANLAB_EXPERIMENT_NAME}"
    printf 'slime_ckpt=%s\n' "${SLIME_CKPT}"
    printf 'checkpoint_dir=%s/%s\n' "${SLIME_CKPT}" "${ckpt_iter_dir}"
    date -Iseconds | sed 's/^/completed_at=/'
  } > "${done_marker}"
  echo "Checkpoint ${step} full-valid eval complete."
}

main() {
  preflight
  maybe_run_or_wait_training
  require_path "${SLIME_CKPT}" "SFT checkpoint root"
  init_sweep_state
  generate_full_eval_indices

  echo "SwanLab sweep id: ${SWEEP_ID}"
  echo "SwanLab sweep run_id: ${SWANLAB_RUN_ID}"
  echo "Metrics will be logged with SwanLab step = checkpoint step."

  local steps
  steps=$(selected_ckpt_steps)
  if [[ -z "${steps}" ]]; then
    echo "ERROR: no checkpoint iter_* directories found under ${SLIME_CKPT}" >&2
    exit 2
  fi
  printf '%s\n' "${steps}" > "${SWEEP_STATE_DIR}/checkpoint_steps.txt"
  echo "Checkpoint steps for this sweep are recorded at ${SWEEP_STATE_DIR}/checkpoint_steps.txt"

  local idx=0
  local step
  while IFS= read -r step; do
    [[ -n "${step}" ]] || continue
    start_ray_for_step "${idx}" "${step}"
    local status=0
    set +e
    run_eval_step "${step}"
    status=$?
    set -e
    stop_active_ray
    if [[ "${status}" -ne 0 ]]; then
      exit "${status}"
    fi
    idx=$((idx + 1))
  done <<< "${steps}"

  echo "All checkpoint full-valid evaluations finished in one SwanLab run: ${SWANLAB_RUN_ID}"
}

main "$@"
