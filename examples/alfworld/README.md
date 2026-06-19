# ALFWorld GRPO / GRPO+OPSD / OPSD

This example trains `Qwen2.5-3B-Instruct` with slime GRPO in the ALFWorld TextWorld environment, and can optionally add OPSD-style privileged teacher scoring on top of slime's native OPD advantage penalty. It also includes a pure OPSD launcher that zeros processed task rewards while keeping raw ALFWorld rewards for metrics. It uses `--rollout-function-path batched_rollout.generate_rollout` so one rollout batch can coordinate many active agent-environment episodes: model action requests are issued concurrently per environment step, and ALFWorld environment state is kept in Ray actors.

The GRPO script is intended to reproduce the ALFWorld GRPO experiment from the SDAR paper [Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155) by Meituan and Zhejiang University. The GRPO+OPSD script keeps the same ALFWorld rollout and reward path, but when `--use-opd --opd-type self` is enabled it scores each student step response under SDAR-style privileged ALFWorld skills on the current rollout SGLang router and passes `teacher_log_probs` to slime's existing OPD machinery. The pure OPSD script uses the same teacher-logprob path but sets processed training rewards to `0.0`, so the OPD term is the policy signal.

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
ALFWORLD_MAX_STEPS=50 \
bash examples/alfworld/run_qwen2.5_3B_instruct_grpo.sh
```

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

## How the example works

- `prepare_alfworld_data.py` scans `$ALFWORLD_DATA/json_2.1.1/{train,valid_seen,valid_unseen}` and keeps solvable `game.tw-pddl` tasks.
- `batched_rollout.py` resets one `AlfredTWEnv` episode per trajectory in Ray actors, prompts the model with the current observation and admissible actions, parses `<think>...</think><action>...</action>`, steps all active environments in parallel, and returns step-level slime `Sample` objects.
- `generate_with_alfworld.py` keeps the single-episode fallback implementation plus shared reward/filter/helper functions used by the batched rollout path.
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
| `batched_rollout.py` | custom batched ALFWorld rollout function used by `--rollout-function-path` |
| `opsd.py` | SDAR-style privileged skill loading and teacher log-prob scoring helpers |
| `skills/` | ALFWorld privileged skill mapping and markdown copied from SDAR runtime skills |
| `generate_with_alfworld.py` | single-episode fallback plus reward/filter/helper functions |
| `alfworld_env.py` | lazy ALFWorld TextWorld episode wrapper |
| `prompts.py` | ALFWorld prompt templates and action parser |
| `prepare_alfworld_data.py` | builds JSONL task indices from ALFWorld data |
| `configs/config_tw.yaml` | TextWorld ALFWorld config using `$ALFWORLD_DATA` |
| `requirements.txt` | optional ALFWorld dependency set |
