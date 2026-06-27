# WebShop GRPO example

This example trains `Qwen2.5-3B-Instruct` with slime GRPO against a separately deployed WebShop HTTP service. The defaults target the WebShop small synthetic setup: `env_seed=0`, `max_steps=15`, train pool `goal_idx >= 500`, validation pool `goal_idx < 500`, `train_batch_size=16`, `rollout.n=8`, `val_batch_size=128`, `test_freq=5`, and `total_epochs=150`.

## Files

| File | Purpose |
| --- | --- |
| `client.py` | Async client for the WebShop HTTP service. |
| `prompts.py` | Prompt template and action parser. |
| `generate_with_webshop.py` | Custom slime generation function, GRPO reward normalization, rollout/eval metrics. |
| `prepare_webshop_data.py` | Builds lightweight WebShop goal metadata JSONL files: `train.jsonl` and `valid.jsonl`. |
| `run_qwen2.5_3B_instruct_grpo.sh` | Qwen2.5-3B-Instruct GRPO launcher with WebShop small synthetic defaults. |

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
- `/root/slime-webshop/valid.jsonl` — validation batch of 128 prompt groups.
- `/root/slime-webshop/summary.json`.

The generated schedule samples validation goals from `[0, 500)`, training goals from `[500, goal_count)`, and each prompt group is repeated by `N_SAMPLES_PER_PROMPT=8` during rollout.

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
