# WebShop GRPO example

This example trains `Qwen2.5-3B-Instruct` with slime GRPO against a separately deployed WebShop HTTP service. The defaults target the WebShop small synthetic setup: `env_seed=0`, `max_steps=15`, train pool `goal_idx >= 500`, train-time validation `goal_idx < 100`, full held-out validation `goal_idx < 500`, `train_batch_size=16`, `rollout.n=8`, `val_batch_size=100`, `test_freq=5`, and `total_epochs=150`.

## Files

| File | Purpose |
| --- | --- |
| `client.py` | Async client for the WebShop HTTP service. |
| `prompts.py` | Prompt template and action parser. |
| `generate_with_webshop.py` | Custom slime generation function, GRPO reward normalization, rollout/eval metrics. |
| `prepare_webshop_data.py` | Builds lightweight WebShop goal metadata JSONL files: `train.jsonl` and `valid.jsonl`. |
| `run_qwen2.5_3B_instruct_grpo.sh` | Qwen2.5-3B-Instruct GRPO launcher with WebShop small synthetic defaults. |
| `eval_qwen2.5_3B_instruct_full_valid.sh` | Eval-only launcher for the full 500-goal held-out validation pool. |
| `eval_all_checkpoints_full_valid.sh` | Eval-only sweep over every saved checkpoint; logs all full-valid scores into one SwanLab run with checkpoint step as the x-axis. |
| `checkpoint_eval_logger.py` | Custom eval logger used by the checkpoint sweep to force `eval/step = CKPT_STEP`. |
| `SEARCH_INDEX_TROUBLESHOOTING.md` | Postmortem and runbook for the failure where `success_rate` stayed at 0 because `indexes_1k` was empty or stale. |

## 1. Start the WebShop service

In the modified WebShop service repo after completing the normal small data/index setup:

```bash
cd ../WebShop
./setup.sh -d small
PORT=3001 SEED=0 ./run_webshop_service.sh
```

Useful service endpoints:

- `GET /health`
- `GET /v1/goals?limit=0`
- `GET /v1/goals?goal_seed=7&limit=5`
- `POST /v1/reset` with optional `goal_idx`, `goal_seed`, `observation_mode`
- `POST /v1/step`
- `DELETE /v1/session/<session_id>`

Check the small synthetic pool:

```bash
curl -s 'http://127.0.0.1:3001/v1/goals?limit=0' | jq
```

The expected small synthetic setup reports `goal_count: 6910`.

## 2. Prepare slime prompt data

```bash
cd ../slime
python examples/webshop/prepare_webshop_data.py \
  --service-url http://127.0.0.1:3001 \
  --output-dir /root/slime-webshop
```

This writes the fixed slime WebShop task schedule:

- `/root/slime-webshop/train.jsonl` — 150 rollout batches × 16 prompt groups; metadata includes `goal_idx` and worker `goal_seed`.
- `/root/slime-webshop/valid.jsonl` — train-time validation batch of 100 prompt groups, `goal_idx` 0 through 99.
- `/root/slime-webshop/summary.json`.

The generated schedule keeps validation goals from `[0, 100)`, training goals from `[500, goal_count)`, and each prompt group is repeated by `N_SAMPLES_PER_PROMPT=8` during rollout. Goals `[100, 500)` are held out from training-time eval and are included only in the full validation launcher below.

## 3. Launch GRPO

```bash
MODEL_ROOT=/root/Qwen2.5-3B-Instruct \
MCORE_CKPT=/root/Qwen2.5-3B-Instruct_torch_dist \
SLIME_CKPT=/root/Qwen2.5-3B-Instruct_webshop_grpo_slime \
WEBSHOP_TASK_DIR=/root/slime-webshop \
WEBSHOP_SERVICE_URL=http://127.0.0.1:3001 \
bash examples/webshop/run_qwen2.5_3B_instruct_grpo.sh
```

Important launcher defaults:

- `ROLLOUT_BATCH_SIZE=16`
- `N_SAMPLES_PER_PROMPT=8`
- `GLOBAL_BATCH_SIZE=128`
- `NUM_ROLLOUT=150`
- `EVAL_INTERVAL=5`, with eval before train enabled by slime default
- rollout internals use `max_steps=15`, `history_length=4` (override with `WEBSHOP_HISTORY_LENGTH`), per-step generation cap `512`, history fallback threshold `13000` chars, invalid-action penalty `0.1` on the invalid step
- `MAX_TOKENS_PER_GPU=32768`
- `LOG_PROBS_CHUNK_SIZE=8192`
- eval samples per prompt `1`, temperature `0.4`, top-p `1.0`
- single eval dataset: `valid.jsonl`

## 4. Run full held-out validation only

Use the full-valid eval launcher after training when you want to score a saved checkpoint on all 500 held-out WebShop goals. The launcher writes a separate full-eval task directory and does not overwrite the training-time files under `/root/slime-webshop`.

```bash
MODEL_ROOT=/root/Qwen2.5-3B-Instruct \
SLIME_CKPT=/root/Qwen2.5-3B-Instruct_webshop_grpo_slime \
WEBSHOP_FULL_EVAL_TASK_DIR=/root/slime-webshop-full-eval \
WEBSHOP_SERVICE_URL=http://127.0.0.1:3001 \
bash examples/webshop/eval_qwen2.5_3B_instruct_full_valid.sh
```

By default the script loads the latest checkpoint recorded by `latest_checkpointed_iteration.txt`. To evaluate a specific saved checkpoint, set `CKPT_STEP`; for example, `CKPT_STEP=50` loads `iter_0000050` from `SLIME_CKPT`.

The script runs slime in eval-only mode with `--num-rollout 0` and `--eval-interval 1`, generates `valid_full.jsonl` with `goal_idx` 0 through 499, logs metrics under `eval/valid_full/...`, and exits without training.


## 5. Sweep all checkpoints on full validation

Use the all-checkpoint full-valid launcher when you want one SwanLab experiment containing every saved checkpoint's full 500-goal validation score. The sweep scans `SLIME_CKPT/iter_*` by default, evaluates each checkpoint serially, and passes the same `SWANLAB_RUN_ID` to every eval-only run. The custom logger writes `eval/step` as the checkpoint step, so SwanLab's x-axis is the checkpoint number rather than the eval rollout id.

```bash
MODEL_ROOT=/data/zhangdw12/models/Qwen2.5-3B-Instruct \
MCORE_CKPT=/data/zhangdw12/models/Qwen2.5-3B-Instruct_torch_dist \
SLIME_CKPT=/data/zhangdw12/models/Qwen2.5-3B-Instruct_webshop_grpo_slime \
WEBSHOP_SERVICE_URL=http://superagent-ai02:3001 \
WEBSHOP_FULL_EVAL_TASK_DIR=/data/zhangdw12/datasets/slime-webshop-full-eval \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_GPUS=4 \
bash examples/webshop/eval_all_checkpoints_full_valid.sh
```

To evaluate only selected checkpoints, pass comma-separated steps:

```bash
CKPT_STEPS=10,20,30 bash examples/webshop/eval_all_checkpoints_full_valid.sh
```

To resume a failed sweep into the same SwanLab run, reuse the printed `SWEEP_ID` or `SWANLAB_RUN_ID`. Done markers and local logs are stored under `${SLIME_CKPT}/all_ckpt_full_eval_tracking/` by default; `SKIP_EXISTING=1` skips checkpoints already completed for the same sweep/run id.
