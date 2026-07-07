# WebShop GRPO example

This example trains `Qwen2.5-3B-Instruct` with slime GRPO against a separately deployed WebShop HTTP service. The defaults target the WebShop small synthetic setup: `env_seed=0`, `max_steps=15`, train pool `goal_idx >= 500`, full train-time validation `goal_idx < 500`, `rollout_batch_size=16`, `rollout.n=8`, `test_freq=5`, and `NUM_ROLLOUT=150` in the launcher.

## Files

| File | Purpose |
| --- | --- |
| `client.py` | Async client for the WebShop HTTP service. |
| `prompts.py` | Prompt template and action parser. |
| `generate_with_webshop.py` | Custom slime generation function, GRPO reward normalization, rollout/eval metrics. |
| `prepare_webshop_data.py` | Builds lightweight WebShop goal metadata JSONL files: `train.jsonl` and `valid.jsonl`. |
| `run_qwen2.5_3B_instruct_grpo.sh` | Qwen2.5-3B-Instruct GRPO launcher with WebShop small synthetic defaults. |
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

This writes the fixed slime WebShop task dataset:

- `/root/slime-webshop/train.jsonl` — full small train pool, `goal_idx` 500 through 6909 by default; metadata includes `goal_idx` and `goal_seed`.
- `/root/slime-webshop/valid.jsonl` — full train-time validation pool of 500 prompt groups, `goal_idx` 0 through 499.
- `/root/slime-webshop/summary.json`.

The generated dataset is independent of training length: validation goals cover the full held-out pool `[0, 500)`, training goals come from the full `[500, goal_count)` pool, and each training prompt group is repeated by `N_SAMPLES_PER_PROMPT=8` during rollout. Change `NUM_ROLLOUT` to run more or fewer training steps without regenerating a differently sized dataset.

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
- rollout internals use `max_steps=15`, `history_length=4` (override with `WEBSHOP_HISTORY_LENGTH`), per-step generation cap `512`, history fallback threshold `13000` chars, native-scale reward (`dense`: raw score in `0..1`; `binary`: full-success `0/1`), and trajectory-level invalid-action penalty `0.01 * invalid_action_count` before GRPO normalization
- `MAX_TOKENS_PER_GPU=32768`
- `LOG_PROBS_CHUNK_SIZE=8192`
- eval samples per prompt `1`, temperature `0.4`, top-p `1.0`
- single eval dataset: `valid.jsonl`

Training and eval logs include WebShop paper-parity metrics from the same episode summary function:

- `webshop/score`: mean raw WebShop task score with partial credit, matching the usual paper `score`/task-score definition before any 0-100 table scaling.
- `webshop/succ`: harsh full-success rate, counted only when the episode is done and raw task score reaches `1.0`.

`webshop/final_reward_mean` is the training reward after reward-mode conversion and invalid-action penalty, while `webshop/raw_reward_mean` remains the raw task score. The legacy `webshop/success_rate` still counts any positive raw score. Eval prefixes these as `eval/<dataset_name>/webshop/...`, for example `eval/valid/webshop/score` and `eval/valid/webshop/succ`.
