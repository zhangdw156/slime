# WebShop GRPO example

This example trains `Qwen2.5-3B-Instruct` with slime GRPO against a separately deployed WebShop HTTP service. WebShop's heavier runtime dependencies stay in the service repo; slime workers only use the thin HTTP client in this directory.

## Files

| File | Purpose |
| --- | --- |
| `client.py` | Async client for the WebShop HTTP service. |
| `prompts.py` | Prompt template, action parser, and safe fallback action logic. |
| `generate_with_webshop.py` | Custom slime generation function, dynamic filter, GRPO reward normalization, rollout/eval metrics. |
| `prepare_webshop_data.py` | Builds lightweight `train.jsonl`, `valid_seen.jsonl`, and `valid_unseen.jsonl` goal-index files. |
| `run_qwen2.5_3B_instruct_grpo.sh` | Formal Qwen2.5-3B-Instruct GRPO launcher. |

## 1. Start the WebShop service

In the WebShop service repo after completing the normal WebShop data/index setup:

```bash
PORT=3001 NUM_PRODUCTS=1000 ./run_webshop_service.sh
```

Useful service endpoints:

- `GET /health`
- `GET /v1/goals?limit=0`
- `POST /v1/reset`
- `POST /v1/step`
- `DELETE /v1/session/<session_id>`

If Ray workers run on a different host/container from the service, set `WEBSHOP_SERVICE_URL` to a reachable host/IP instead of `127.0.0.1`.

## 2. Prepare slime prompt data

Generate stable WebShop goal-index JSONL files:

```bash
python examples/webshop/prepare_webshop_data.py \
  --service-url http://127.0.0.1:3001 \
  --output-dir /root/slime-webshop \
  --shuffle
```

This writes:

- `/root/slime-webshop/train.jsonl`
- `/root/slime-webshop/valid_seen.jsonl`
- `/root/slime-webshop/valid_unseen.jsonl`
- `/root/slime-webshop/all.jsonl`
- `/root/slime-webshop/summary.json`

Each row keeps the actual instruction inside the WebShop service and stores only `metadata.goal_idx` in slime.

## 3. Launch GRPO

Configure paths and run:

```bash
MODEL_ROOT=/root/Qwen2.5-3B-Instruct \
MCORE_CKPT=/root/Qwen2.5-3B-Instruct_torch_dist \
SLIME_CKPT=/root/Qwen2.5-3B-Instruct_webshop_grpo_slime \
WEBSHOP_TASK_DIR=/root/slime-webshop \
WEBSHOP_SERVICE_URL=http://127.0.0.1:3001 \
bash examples/webshop/run_qwen2.5_3B_instruct_grpo.sh
```

The launcher uses:

- `--custom-generate-function-path generate_with_webshop.generate`
- `--dynamic-sampling-filter-path generate_with_webshop.check_episode_reward_nonzero_std`
- `--custom-reward-post-process-path generate_with_webshop.grpo_normalize_webshop_steps`
- `--custom-rollout-log-function-path generate_with_webshop.log_webshop_rollout`
- `--custom-eval-rollout-log-function-path generate_with_webshop.log_webshop_eval_rollout`

## Runtime knobs

| Variable | Default | Meaning |
| --- | ---: | --- |
| `WEBSHOP_MAX_STEPS` | `30` | Max environment actions per episode. |
| `WEBSHOP_HISTORY_LENGTH` | `4` | Recent observation/action pairs in the prompt. |
| `WEBSHOP_STEP_MAX_TOKENS` | `256` | Per-action generation cap. |
| `WEBSHOP_MAX_PROMPT_CHARS` | `12000` | Prompt char cap before history is dropped. |
| `WEBSHOP_INVALID_ACTION_PENALTY` | `0.0` | Penalty subtracted per invalid model action. |
| `WEBSHOP_OBSERVATION_MODE` | `text` | Service observation mode. |
| `WEBSHOP_HTTP_RETRIES` | `10` | HTTP retries for service calls. |
| `WEBSHOP_CLOSE_SESSION_ON_DONE` | `1` | Close service sessions after each rollout. |

The rollout emits one trainable `Sample` per model action and broadcasts the final WebShop episode reward to all steps in the same trajectory. The custom reward post-process normalizes one reward per trajectory inside each prompt group, then broadcasts the normalized value back to every step so longer episodes are not over-weighted just because they contain more actions.
