# WebShop search-index troubleshooting: `success_rate` stays at 0

This note records a real failure mode where WebShop GRPO evaluation stayed at
zero even though the WebShop HTTP service was reachable and the prompt format was
valid. The final root cause was an empty or broken `search_engine/indexes_1k`
Lucene index in the WebShop service repo.

## Bottom line

If WebShop rollout/eval repeatedly produces trajectories like:

```text
search[...] -> click[next >] -> click[back to search] -> ... -> reward=0.0
```

and no trajectory ever reaches `click[buy now]`, do not stop at `/health`. Check
whether the WebShop search index actually returns products. In the observed
failure, `/health` reported a healthy service with 1000 products, but every
`search[...]` returned `Page 1 (Total results: 0)` because `indexes_1k` had not
been built correctly.

## Symptoms we saw

- `eval/valid/webshop/success_rate` stayed at `0`.
- Eval examples had only `reward=0.0`.
- There were many `search[...]`, `click[next >]`, and `click[back to search]`
  actions, but zero `click[buy now]` actions.
- After prompt guidance was improved, the model started issuing shorter search
  queries such as `search[dress shirts]`, but the service still returned zero
  search results.
- Direct service checks also returned zero results for broad queries such as
  `shirt`, `shoe`, `black`, `blue`, and `men`.

This means the model could not enter a product page, so reward and success-rate
metrics could not improve.

## What was not the root cause

- **Not just prompt formatting.** The prompt change from `search[keywords]` to
  `search[<your query>]` and the short-query guidance made the model search more
  reasonably, but search still returned no products.
- **Not service reachability.** `/health` only proves the Flask service is alive
  and has loaded goals/products. It does not prove Lucene search returns hits.
- **Not `goal_idx` alone.** Slime WebShop samples carry both `goal_idx` and
  `goal_seed`; exact reproduction of eval examples should pass both.

## Exact diagnosis path

### 1. Confirm the prompt is not the active issue

A healthy prompt should advertise `search[<your query>]` and short-query guidance:

```bash
grep -n "search\\[<your query>\\]\|short core product query\|Do not put color" \
  /data/zhangdw12/work/slime/examples/webshop/prompts.py
```

If eval examples still start with full-instruction searches containing color,
size, price, and every attribute, the active rollout process may be using an old
prompt or an old worker. Restart the job after updating code.

### 2. Reproduce one eval goal through the service

For validation rows, `goal_seed` is normally `1000 + env_id`; for `goal_idx=49`,
that is `1049` in the default schedule.

```bash
SERVICE=http://superagent-ai02:3001
SID=debug-webshop-$(date +%s)

curl -s -X POST "$SERVICE/v1/reset" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"goal_idx\":49,\"goal_seed\":1049}" | \
  jq -r '.instruction_text, .observation'

curl -s -X POST "$SERVICE/v1/step" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"action\":\"search[dress shirts]\"}" | \
  jq -r '.observation, .available_actions'

curl -s -X DELETE "$SERVICE/v1/session/$SID" >/dev/null
```

If this returns `Page 1 (Total results: 0)`, the failure is not just a rollout log
artifact.

### 3. Check broad search terms through the service

```bash
SERVICE=http://superagent-ai02:3001
SID=debug-webshop-$(date +%s)

curl -s -X POST "$SERVICE/v1/reset" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"goal_idx\":0,\"goal_seed\":1000}" >/dev/null

for q in "shirt" "shoe" "black" "blue" "case" "table" "phone" "women" "men"; do
  echo "=== $q ==="
  curl -s -X POST "$SERVICE/v1/step" \
    -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"$SID\",\"action\":\"search[$q]\"}" | jq -r '.observation'
done

curl -s -X DELETE "$SERVICE/v1/session/$SID" >/dev/null
```

If broad terms all return `Total results: 0`, the search backend is effectively
non-functional for the running service.

### 4. Check the WebShop service process

```bash
ps -ef | grep -E 'webshop|WebShop|run_webshop_service|3001' | grep -v grep

PID=$(pgrep -f 'web_agent_site.service.api.*--port 3001' | head -1)
tr '\0' '\n' < /proc/$PID/environ | grep -E 'WEBSHOP|DATA|INDEX|NUM_PRODUCTS|PORT'
```

The default 1k service normally runs with `NUM_PRODUCTS=1000` and loads:

```text
data/items_shuffle_1000.json
data/items_ins_v2_1000.json
search_engine/indexes_1k
```

### 5. Check Lucene directly

Run this from the WebShop repo, not the slime repo:

```bash
cd /data/zhangdw12/work/WebShop

python - <<'PY'
import json
from pyserini.search.lucene import LuceneSearcher

items = json.load(open('data/items_shuffle_1000.json'))
asins = {x['asin'] for x in items[:1000]}

s = LuceneSearcher('search_engine/indexes_1k')
for q in ['shirt', 'shoe', 'black', 'blue', 'men']:
    hits = s.search(q, k=10)
    ids = []
    for h in hits:
        raw = json.loads(s.doc(h.docid).raw())
        ids.append(raw.get('id'))
    print('QUERY', q)
    print('hits', ids)
    print('in_current_items', [x for x in ids if x in asins])
PY
```

Interpretation:

- `hits []`: `indexes_1k` is empty, missing, or corrupted.
- `hits` non-empty but `in_current_items []`: the index and the current
  `items_shuffle_1000.json` do not match.
- both non-empty: Lucene and the 1k corpus are aligned; check whether the running
  service is using a different repo/path or an old in-memory index.

In the observed incident, `search_engine/resources_1k/documents.jsonl` did not
exist and `search_engine/indexes_1k` was only about `8K`, so all direct Lucene
searches returned `hits []`.

## Fix: rebuild resources and `indexes_1k`

Run in the WebShop repo:

```bash
cd /data/zhangdw12/work/WebShop/search_engine

mkdir -p resources resources_100 resources_1k resources_100k
python convert_product_file_format.py

wc -l resources_1k/documents.jsonl
head -1 resources_1k/documents.jsonl
```

`resources_1k/documents.jsonl` should have about 1000 lines.

Then rebuild only the 1k Lucene index:

```bash
cd /data/zhangdw12/work/WebShop/search_engine

mv indexes_1k indexes_1k.bad.$(date +%s) 2>/dev/null || true

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input resources_1k \
  --index indexes_1k \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw
```

Re-run the direct Lucene check. A successful result should show non-empty
`hits`, and those ASINs should also appear under `in_current_items`.

## Required restart after rebuilding

The WebShop service loads the search engine at startup. Rebuilding the index on
disk is not enough; restart the service so it picks up the new index.

```bash
pkill -f 'web_agent_site.service.api.*--port 3001'

cd /data/zhangdw12/work/WebShop
HOST=superagent-ai02 PORT=3001 NUM_PRODUCTS=1000 bash ./run_webshop_service.sh
```

Then check service health and service search:

```bash
curl -s http://superagent-ai02:3001/health | jq

SERVICE=http://superagent-ai02:3001
SID=debug-webshop-$(date +%s)

curl -s -X POST "$SERVICE/v1/reset" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"goal_idx\":0,\"goal_seed\":1000}" >/dev/null

curl -s -X POST "$SERVICE/v1/step" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"action\":\"search[men]\"}" | \
  jq -r '.observation, .available_actions'

curl -s -X DELETE "$SERVICE/v1/session/$SID" >/dev/null
```

The service search output should no longer say `Total results: 0` for broad
queries, and `available_actions.clickables` should include product entries in
addition to navigation buttons.

## Training guidance after the fix

Stop and rerun any training job that already rolled out against the empty index.
Those rollouts contain no useful product-page or purchase trajectories.

Recommended order:

1. Rebuild `resources_1k` and `indexes_1k`.
2. Verify direct Lucene hits.
3. Restart the WebShop service.
4. Verify service-level search returns products.
5. Restart the slime WebShop GRPO run.
6. Watch eval examples for:
   - shorter `search[...]` queries,
   - product click actions after search,
   - eventual `click[buy now]`,
   - non-zero raw reward or success-rate movement.

## Key lesson

A healthy WebShop `/health` response is necessary but not sufficient. For WebShop
GRPO, always include a search-index smoke test before trusting eval metrics:

```text
/health ok + goal_count ok + Lucene hits ok + service search returns products
```
