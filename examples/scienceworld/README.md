# ScienceWorld GRPO

This example trains `Qwen2.5-3B-Instruct` with slime GRPO in the ScienceWorld text environment. It uses `--rollout-function-path batched_rollout.generate_rollout` so one rollout batch can coordinate many active ScienceWorld episodes: model action requests are issued concurrently per environment step, and mutable Java environment instances are kept in Ray actors.

## 1. Environment setup

Start from a working slime environment, then install Java and the optional ScienceWorld dependency:

```bash
cd /root/slime
pip install -e . --no-deps
mamba install -c conda-forge openjdk=11
pip install -r examples/scienceworld/requirements.txt
```

If your ScienceWorld installation uses a custom jar, export `SCIENCEWORLD_JAR_PATH=/path/to/scienceworld.jar` before data preparation and training.

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

## 3. Prepare ScienceWorld task/variation data

```bash
python examples/scienceworld/prepare_scienceworld_data.py \
  --local-dir /root/slime-scienceworld \
  --simplification easy \
  --env-step-limit 100 \
  --train-size -1 \
  --eval-size 32 \
  --test-size 32 \
  --shuffle
```

The output files are:

```text
/root/slime-scienceworld/train_tasks.jsonl
/root/slime-scienceworld/eval_tasks.jsonl
/root/slime-scienceworld/test_tasks.jsonl
```

Each row stores an `index` prompt plus `metadata.task_name`, `metadata.variation_idx`, `metadata.simplification`, and `metadata.env_step_limit`; the rollout function uses this metadata to load the actual ScienceWorld episode. Use `--tasks` with task names or zero-based task ids to restrict the generated index.

## 4. Run GRPO

```bash
cd /root/slime
bash examples/scienceworld/run_qwen2.5_3B_instruct_grpo.sh
```

Useful overrides:

```bash
NUM_GPUS=4 TP_SIZE=1 \
ROLLOUT_BATCH_SIZE=8 N_SAMPLES_PER_PROMPT=8 \
SCIENCEWORLD_TASK_DIR=/root/slime-scienceworld \
SCIENCEWORLD_SIMPLIFICATION=easy \
SCIENCEWORLD_STEP_MAX_TOKENS=512 \
SCIENCEWORLD_MAX_STEPS=50 \
SCIENCEWORLD_ENV_WORKER_CPUS=0.25 \
SCIENCEWORLD_ENV_WORKER_MAX_EPISODES=1 \
bash examples/scienceworld/run_qwen2.5_3B_instruct_grpo.sh
```

`SCIENCEWORLD_ENV_WORKER_MAX_EPISODES` defaults to `1`, retiring each Ray environment actor after one episode for conservative long-run Java process hygiene. Increase it if environment startup overhead dominates and your run is stable.

## How the example works

- `prepare_scienceworld_data.py` queries ScienceWorld task names and official train/dev/test variation splits, then writes lightweight JSONL indices.
- `batched_rollout.py` resets one ScienceWorld episode per trajectory in Ray actors, prompts the model with task description, observation, inventory, and admissible actions, parses `<think>...</think><action>...</action>`, steps all active environments in parallel, and returns step-level slime `Sample` objects.
- `generate_with_scienceworld.py` keeps the single-episode fallback implementation plus shared reward/filter/helper functions used by the batched rollout path.
- The reward is `final_score - SCIENCEWORLD_INVALID_ACTION_PENALTY * invalid_action_count`; the default invalid-action penalty is `0.01`.
- `loss_mask` is `1` only on assistant-generated tokens and `0` on environment/user-observation tokens.
- GRPO reward normalization is done by `generate_with_scienceworld.grpo_normalize_scienceworld_steps`, which normalizes once per trajectory and broadcasts the normalized value to that trajectory's step samples.

## File map

| File | Purpose |
| --- | --- |
| `run_qwen2.5_3B_instruct_grpo.sh` | slime launch script for Qwen2.5-3B-Instruct GRPO |
| `batched_rollout.py` | custom batched ScienceWorld rollout function used by `--rollout-function-path` |
| `generate_with_scienceworld.py` | single-episode fallback plus reward/filter/helper functions |
| `scienceworld_env.py` | lazy ScienceWorld episode wrapper |
| `prompts.py` | ScienceWorld prompt templates and action parser |
| `prepare_scienceworld_data.py` | builds JSONL task/variation indices from ScienceWorld splits |
| `requirements.txt` | optional ScienceWorld dependency |
