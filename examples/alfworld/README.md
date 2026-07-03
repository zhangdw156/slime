# ALFWorld GRPO / GRPO+OPSD / OPSD / SFT Distillation

This example trains `Qwen2.5-3B-Instruct` with slime GRPO in the ALFWorld TextWorld environment, and can optionally add OPSD-style privileged teacher scoring on top of slime's native OPD advantage penalty. It also includes a pure OPSD launcher that zeros processed task rewards while keeping raw ALFWorld rewards for metrics, plus a `Qwen2.5-0.5B-Instruct` SFT launcher for distilling successful 3B teacher trajectories into a smaller student. It uses `--rollout-function-path batched_rollout.generate_rollout` so one rollout batch can coordinate many active agent-environment episodes: model action requests are issued concurrently per environment step, and ALFWorld environment state is kept in Ray actors.

The GRPO script is intended to reproduce the ALFWorld GRPO experiment from the SDAR paper [Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155) by Meituan and Zhejiang University. The GRPO+OPSD script keeps the same ALFWorld rollout and reward path, but when `--use-opd --opd-type self` is enabled it scores each student step response under SDAR-style privileged ALFWorld skills on the current rollout SGLang router and passes `teacher_log_probs` to slime's existing OPD machinery. The pure OPSD script uses the same teacher-logprob path but sets processed training rewards to `0.0`, so the OPD term is the policy signal.

Unless a launcher or experiment note explicitly marks an ablation/comparison setting, every experiment under `examples/alfworld` uses `ALFWORLD_HISTORY_LENGTH=4` by default. Keep `ALFWORLD_HISTORY_LENGTH=4` for standard training, SFT, OPD/OPSD, and full-valid evaluation runs; use another value only for intentionally named history-length ablations.

## 1. Environment setup

Start from a working slime environment, then install the ALFWorld dependencies:

```bash
cd /root/slime
pip install -e . --no-deps
pip install -r examples/alfworld/requirements.txt
alfworld-download -f
```

By default ALFWorld downloads data under `~/.cache/alfworld`. If you keep it elsewhere, export `ALFWORLD_DATA` before data preparation and training.

## 2. Prepare model checkpoints

```bash
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/Qwen2.5-3B-Instruct

cd /root/slime
source scripts/models/qwen2.5-3B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-3B-Instruct \
    --save /root/Qwen2.5-3B-Instruct_torch_dist
```

For the 0.5B SFT student launcher, also prepare the 0.5B checkpoint:

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/Qwen2.5-0.5B-Instruct

cd /root/slime
source scripts/models/qwen2.5-0.5B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-0.5B-Instruct \
    --save /root/Qwen2.5-0.5B-Instruct_torch_dist
```

## 3. Prepare ALFWorld game-index data

```bash
export ALFWORLD_DATA=${ALFWORLD_DATA:-/root/.cache/alfworld}
python examples/alfworld/prepare_alfworld_data.py \
  --alfworld-data "$ALFWORLD_DATA" \
  --local-dir /root/slime-alfworld \
  --train-size -1 \
  --valid-seen-size 32 \
  --valid-unseen-size 32
```

The output files are:

```text
/root/slime-alfworld/train_games.jsonl
/root/slime-alfworld/valid_seen_games.jsonl
/root/slime-alfworld/valid_unseen_games.jsonl
```

Each row stores an `index` prompt plus `metadata.gamefile`; the rollout function uses the gamefile to load the actual ALFWorld episode.

## 4. Run GRPO, GRPO+OPSD, or pure OPSD

```bash
cd /root/slime
bash examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh
```

Useful overrides:

```bash
NUM_GPUS=4 TP_SIZE=2 \
ROLLOUT_BATCH_SIZE=16 N_SAMPLES_PER_PROMPT=8 \
ALFWORLD_TASK_DIR=/root/slime-alfworld \
ALFWORLD_DATA=/root/.cache/alfworld \
ALFWORLD_STEP_MAX_TOKENS=512 \
ALFWORLD_ENV_WORKER_CPUS=0.1 \
ALFWORLD_ENV_WORKER_MAX_EPISODES=1 \
ALFWORLD_MAX_STEPS=50 \
bash examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh
```

`ALFWORLD_ENV_WORKER_MAX_EPISODES` defaults to `1` so each Ray environment
actor is retired after one ALFWorld episode. This avoids long-running
TextWorld/Fast-Downward native-library accumulation without doing unsafe
in-process `dlclose`/reload cycles.

To run the GRPO+OPSD variant, use the `grpo_opsd` launcher instead:

```bash
cd /root/slime
bash examples/alfworld/run_qwen2.5_3B_instruct_grpo_opsd.sh
```

To run the pure OPSD variant, use the `_opsd.sh` launcher:

```bash
cd /root/slime
bash examples/alfworld/run_qwen2.5_3B_instruct_opsd.sh
```

Both OPSD launchers default to `OPSD_TYPE=self` and `ALFWORLD_OPSD_SKILLS_DIR=examples/alfworld/skills`. In `self` mode they use the current rollout SGLang router as the teacher scorer with `max_new_tokens=0`, so no `--rm-url` or separately deployed teacher is required. To compare against an external SGLang teacher, set `OPSD_TYPE=sglang` and `ALFWORLD_OPSD_TEACHER_URL=http://teacher-host:port/generate`. The pure OPSD launcher does not enable `--use-kl-loss`, so it avoids an extra reference KL path.

## 5. Build 3B teacher SFT data and train a 0.5B student

After a 3B teacher checkpoint is available, expose it through an SGLang
`/generate` endpoint and collect ALFWorld trajectories. The collector now writes
every sampling attempt to `all_trajectories.jsonl` as soon as that trajectory
finishes. The builder then keeps the shortest successful trajectory per task
and emits one messages-format SFT row per usable step.

```bash
cd /root/slime
python examples/alfworld/collect_teacher_trajectories.py \
  --teacher-url http://127.0.0.1:30000/generate \
  --tokenizer-path /root/Qwen2.5-3B-Instruct \
  --task-file /root/slime-alfworld/train_games.jsonl \
  --output-dir /root/slime-alfworld-teacher-sft \
  --samples-per-task 8 \
  --max-concurrent-tasks 128 \
  --resume

python examples/alfworld/build_sft_from_teacher_trajectories.py \
  --input /root/slime-alfworld-teacher-sft/all_trajectories.jsonl \
  --output /root/slime-alfworld-teacher-sft/alfworld_teacher_sft.jsonl \
  --summary-output /root/slime-alfworld-teacher-sft/build_sft_summary.json
```

`--resume` uses `all_trajectories.jsonl` as the append-only checkpoint ledger,
so already-written failed, truncated, aborted, error, and successful attempts
are all skipped on restart. Each completed trajectory is flushed immediately;
set `--fsync-every N` only if you need an additional durability barrier beyond
normal flushes.


### Inspect collected teacher trajectories

After collection, you can inspect `all_trajectories.jsonl` with the bundled
read-only web viewer:

```bash
cd examples/alfworld/trajectory_viewer
npm run dev
```

Open the printed URL and choose the server directory that contains
`all_trajectories.jsonl`. The viewer streams the JSONL ledger, shows aggregate
analysis such as status counts, success@8, task-type success rates, invalid
actions, token/step averages, grouped sample attempts, per-sample action transition graphs,
shortest-success SFT candidates, and step-by-step trajectory details.

Then train the 0.5B student on the generated messages JSONL:

```bash
cd /root/slime
bash examples/alfworld/run_qwen2.5_0.5B_instruct_sft.sh
```

Useful overrides:

```bash
NUM_GPUS=4 TP_SIZE=1 \
SFT_DATA=/root/slime-alfworld-teacher-sft/alfworld_teacher_sft.jsonl \
MODEL_ROOT=/root/Qwen2.5-0.5B-Instruct \
MCORE_CKPT=/root/Qwen2.5-0.5B-Instruct_torch_dist \
SLIME_CKPT=/root/Qwen2.5-0.5B-Instruct_alfworld_sft_slime \
ALFWORLD_TASK_DIR=/root/slime-alfworld \
USE_EVAL=1 EVAL_INTERVAL=10 \
LOG_PROBS_CHUNK_SIZE=1024 \
bash examples/alfworld/run_qwen2.5_0.5B_instruct_sft.sh
```

Set `USE_EVAL=0` to run training-only SFT with `--debug-train-only`, which
skips SGLang rollout initialization. With `USE_EVAL=1`, the script evaluates the
student on `valid_seen` and `valid_unseen` using the same `batched_rollout.py`
path as the 3B GRPO launcher.

### Native OPD: train a 0.5B student from a trained 3B teacher

If you want slime's framework-native OPD path instead of the ALFWorld
privileged-skill OPSD prompt, first deploy the trained 3B teacher as an SGLang
`/generate` endpoint, then launch the 0.5B student script with `TEACHER_URL`
pointing to that endpoint:

```bash
python3 -m sglang.launch_server \
  --model-path /root/Qwen2.5-3B-Instruct-alfworld-teacher \
  --host 0.0.0.0 \
  --port 30000

cd /root/slime
bash examples/alfworld/run_qwen2.5_0.5B_instruct_opd_from_3B.sh
```

Override `TEACHER_URL`, `MODEL_ROOT`, or `MCORE_CKPT` only when your paths or
teacher endpoint differ from the script defaults. The native OPD launcher
defaults to one rollout per prompt (`N_SAMPLES_PER_PROMPT=1`); increase it only
if you want multiple student trajectories per ALFWorld task. This launcher sets
`ALFWORLD_OPD_USE_NATIVE=1`, uses
`--use-opd --opd-type sglang --rm-url "$TEACHER_URL"`, and reuses
`slime.rollout.on_policy_distillation.reward_func` to score the 0.5B
student's online ALFWorld action tokens under the 3B teacher. It keeps raw
ALFWorld rewards for metrics but returns zero processed training rewards, so
the policy signal is the native OPD term. Unlike the `_opsd.sh` launchers, this
path does not prepend ALFWorld privileged skill text to the teacher prompt.

## 6. Run full valid_seen / valid_unseen evaluation only

Use the full-valid eval launcher when training-time eval used a small exported
subset, such as 32 seen and 32 unseen games, but you want to score a saved
checkpoint on the full ALFWorld `valid_seen` and `valid_unseen` splits. The
launcher regenerates separate full-eval JSONL indices from `$ALFWORLD_DATA` and
does not overwrite the training/eval files under `ALFWORLD_TASK_DIR`.

```bash
cd /root/slime
MODEL_ROOT=/root/Qwen2.5-3B-Instruct \
SLIME_CKPT=/root/Qwen2.5-3B-Instruct_alfworld_grpo_slime \
ALFWORLD_DATA=/root/.cache/alfworld \
ALFWORLD_FULL_EVAL_TASK_DIR=/root/slime-alfworld-full-eval \
bash examples/alfworld/eval_qwen2.5_3B_instruct_full_valid.sh
```

By default the script loads the latest checkpoint recorded by
`latest_checkpointed_iteration.txt`. To evaluate a specific saved checkpoint,
set `CKPT_STEP`; for example, `CKPT_STEP=50` loads `iter_0000050` from
`SLIME_CKPT`:

```bash
CKPT_STEP=50 \
SLIME_CKPT=/root/Qwen2.5-3B-Instruct_alfworld_grpo_slime \
bash examples/alfworld/eval_qwen2.5_3B_instruct_full_valid.sh
```

The script runs slime in eval-only mode with `--num-rollout 0` and
`--eval-interval 1`, so it initializes the model and rollout servers, syncs the
selected checkpoint to SGLang, runs one full evaluation, and exits without
training. Metrics are logged under names such as
`eval/valid_seen_full/alfworld/success_rate` and
`eval/valid_unseen_full/alfworld/success_rate`. SwanLab logging is enabled by
default, matching the training launchers; set `USE_SWANLAB=0` to disable it,
`USE_WANDB=1` to enable W&B, or `USE_TENSORBOARD=1` to enable TensorBoard.

For post-training checkpoint sweeps, use `eval_all_checkpoints_full_valid.sh`.
It scans `SLIME_CKPT/iter_*`, evaluates each saved checkpoint on the same full
valid splits, and logs all results into one SwanLab run with the checkpoint step
as the tracker step. Reuse the printed `SWEEP_ID` to resume a partially finished
sweep without mixing it with a new experiment.

For a Qwen2.5-0.5B SFT checkpoint sweep, override the model-args script and
checkpoint roots while keeping the standard `ALFWORLD_HISTORY_LENGTH=4`. The
SwanLab group, experiment name, and sweep-id prefix are derived from
`MODEL_ARGS_SCRIPT` unless explicitly overridden:

```bash
cd /root/slime
MODEL_ARGS_SCRIPT=qwen2.5-0.5B.sh \
MODEL_ROOT=/root/Qwen2.5-0.5B-Instruct \
MCORE_CKPT=/root/Qwen2.5-0.5B-Instruct_torch_dist \
SLIME_CKPT=/root/Qwen2.5-0.5B-Instruct_alfworld_sft_slime \
bash examples/alfworld/eval_all_checkpoints_full_valid.sh
```

## Docker executable entrypoints

Run these commands from the repository root inside the container, normally
`/root/slime`. All launchers default to Docker-style `/root/...` paths and can
be overridden with environment variables shown above.

| Entrypoint | Command | Requires | Produces / does |
| --- | --- | --- | --- |
| `prepare_alfworld_data.py` | `python examples/alfworld/prepare_alfworld_data.py --alfworld-data /root/.cache/alfworld --local-dir /root/slime-alfworld` | ALFWorld data from `alfworld-download -f` | `train_games.jsonl`, `valid_seen_games.jsonl`, `valid_unseen_games.jsonl` |
| `run_qwen2.5_3B_instruct_grpo.sh` | `bash examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh` | 3B HF + torch_dist checkpoints, prepared game indices | GRPO checkpoint under `SLIME_CKPT` |
| `run_qwen2.5_3B_instruct_grpo_opsd.sh` | `bash examples/alfworld/run_qwen2.5_3B_instruct_grpo_opsd.sh` | same as GRPO plus OPSD skills in `examples/alfworld/skills` | GRPO+OPSD checkpoint under `SLIME_CKPT` |
| `run_qwen2.5_3B_instruct_opsd.sh` | `bash examples/alfworld/run_qwen2.5_3B_instruct_opsd.sh` | same as GRPO+OPSD | pure OPSD checkpoint under `SLIME_CKPT` |
| `collect_teacher_trajectories.py` | `python examples/alfworld/collect_teacher_trajectories.py --teacher-url http://127.0.0.1:30000/generate --tokenizer-path /root/Qwen2.5-3B-Instruct --task-file /root/slime-alfworld/train_games.jsonl --output-dir /root/slime-alfworld-teacher-sft --resume` | running 3B teacher SGLang `/generate` endpoint and prepared train index | realtime `all_trajectories.jsonl` |
| `build_sft_from_teacher_trajectories.py` | `python examples/alfworld/build_sft_from_teacher_trajectories.py --input /root/slime-alfworld-teacher-sft/all_trajectories.jsonl --output /root/slime-alfworld-teacher-sft/alfworld_teacher_sft.jsonl` | collected teacher trajectories | messages-format SFT JSONL |
| `run_qwen2.5_0.5B_instruct_sft.sh` | `bash examples/alfworld/run_qwen2.5_0.5B_instruct_sft.sh` | 0.5B HF + torch_dist checkpoints and SFT JSONL | SFT student checkpoint; optional ALFWorld eval when `USE_EVAL=1` |
| `run_qwen2.5_0.5B_instruct_opd_from_3B.sh` | `bash examples/alfworld/run_qwen2.5_0.5B_instruct_opd_from_3B.sh` | 0.5B HF + torch_dist checkpoints, prepared game indices, trained 3B SGLang `/generate` endpoint | Native slime OPD training of the 0.5B student from the 3B teacher |
| `eval_qwen2.5_3B_instruct_full_valid.sh` | `bash examples/alfworld/eval_qwen2.5_3B_instruct_full_valid.sh` | trained 3B slime checkpoint and full ALFWorld data | full `valid_seen` / `valid_unseen` metrics |
| `eval_all_checkpoints_full_valid.sh` | `bash examples/alfworld/eval_all_checkpoints_full_valid.sh` | trained slime checkpoint root with `iter_*` saves and full ALFWorld data | one SwanLab run containing full-valid metrics for every checkpoint step |

The remaining Python files in this directory (`batched_rollout.py`,
`generate_with_alfworld.py`, `opsd.py`, `alfworld_env.py`, and `prompts.py`) are
imported by the entrypoints above rather than launched directly.

## How the example works

- `prepare_alfworld_data.py` scans `$ALFWORLD_DATA/json_2.1.1/{train,valid_seen,valid_unseen}` and keeps solvable `game.tw-pddl` tasks.
- `batched_rollout.py` resets one `AlfredTWEnv` episode per trajectory in Ray actors, prompts the model with the current observation and admissible actions, parses `<think>...</think><action>...</action>`, steps all active environments in parallel, retires environment actors after a bounded number of episodes, and returns step-level slime `Sample` objects.
- `generate_with_alfworld.py` keeps the single-episode fallback implementation plus shared reward/filter/helper functions used by the batched rollout path.
- `collect_teacher_trajectories.py` samples actions from a running 3B teacher endpoint, defaults to 8 attempts per ALFWorld task, and records every success/failure/truncation/abort/error trajectory in realtime for offline analysis and distillation.
- `build_sft_from_teacher_trajectories.py` selects the shortest successful trajectory for each task from the collected attempts and writes messages-format SFT rows consumed by the 0.5B SFT launcher.
- `run_qwen2.5_0.5B_instruct_sft.sh` uses `slime.rollout.sft_rollout.generate_rollout` with `--loss-type sft_loss`; when `USE_EVAL=1`, it separately uses `batched_rollout.generate_rollout` for ALFWorld eval.
- `opsd.py` mirrors SDAR's ALFWorld privileged skill selection and scores fixed student responses to populate `Sample.teacher_log_probs` when slime OPD is enabled. The default `--opd-type self` path scores on the current rollout router; `OPSD_TYPE=sglang` can point to an external teacher through `ALFWORLD_OPSD_TEACHER_URL`.
- The reward is `1 * won - ALFWORLD_INVALID_ACTION_PENALTY * invalid_action_count`; the default invalid-action penalty is `0.01`.
- `loss_mask` is `1` only on assistant-generated tokens and `0` on environment/user-observation tokens.
- Without `--use-opd`, the training objective is plain GRPO. With `--use-opd --opd-type self`, the rollout also fills `teacher_log_probs` from privileged prompts and slime applies its native OPD advantage penalty. The pure OPSD launcher uses `generate_with_alfworld.zero_alfworld_rewards_for_opsd` to keep `raw_reward` metrics while returning zero processed rewards for training.

## File map

| File | Purpose |
| --- | --- |
| `run_qwen2.5_3B_instruct_grpo.sh` | slime launch script for Qwen2.5-3B-Instruct GRPO |
| `run_qwen2.5_3B_instruct_grpo_opsd.sh` | GRPO launcher with OPSD teacher-logprob scoring enabled via slime OPD |
| `run_qwen2.5_3B_instruct_opsd.sh` | pure OPSD launcher with zero processed rewards and no GRPO reward-std filter |
| `run_qwen2.5_0.5B_instruct_sft.sh` | SFT launcher for distilling 3B teacher ALFWorld data into Qwen2.5-0.5B-Instruct |
| `run_qwen2.5_0.5B_instruct_opd_from_3B.sh` | native slime OPD launcher for online ALFWorld 0.5B rollouts scored by a trained 3B SGLang teacher |
| `eval_qwen2.5_3B_instruct_full_valid.sh` | eval-only launcher that regenerates full valid_seen/valid_unseen indices and logs full-split metrics |
| `eval_all_checkpoints_full_valid.sh` | eval-only checkpoint sweep launcher that logs every `iter_*` full-valid result into one SwanLab run |
| `checkpoint_eval_logger.py` | custom eval logger that uses checkpoint step as the SwanLab/TensorBoard step |
| `collect_teacher_trajectories.py` | collects all teacher trajectory attempts from a running 3B teacher endpoint |
| `build_sft_from_teacher_trajectories.py` | filters successful teacher trajectories into messages-format SFT data |
| `batched_rollout.py` | custom batched ALFWorld rollout function used by `--rollout-function-path` |
| `opsd.py` | SDAR-style privileged skill loading and teacher log-prob scoring helpers |
| `skills/` | ALFWorld privileged skill mapping and markdown copied from SDAR runtime skills |
| `generate_with_alfworld.py` | single-episode fallback plus reward/filter/helper functions |
| `alfworld_env.py` | lazy ALFWorld TextWorld episode wrapper |
| `prompts.py` | ALFWorld prompt templates and action parser |
| `prepare_alfworld_data.py` | builds JSONL task indices from ALFWorld data |
| `configs/config_tw.yaml` | TextWorld ALFWorld config using `$ALFWORLD_DATA` |
| `requirements.txt` | optional ALFWorld dependency set |
