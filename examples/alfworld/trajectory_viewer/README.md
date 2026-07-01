# ALFWorld Teacher Trajectory Viewer

A read-only local multi-page web UI for inspecting `all_trajectories.jsonl` produced by
`examples/alfworld/collect_teacher_trajectories.py`.

## Start

```bash
cd examples/alfworld/trajectory_viewer
npm run dev
```

Then open the printed URL, normally:

```text
http://127.0.0.1:5173
```

Optional environment variables:

```bash
ALFWORLD_VIEWER_HOST=0.0.0.0 ALFWORLD_VIEWER_PORT=5173 npm run dev
```

The viewer is intended for trusted local or SSH-forwarded use. It reads paths on
the machine running `npm run dev`; it does not upload, mutate, or rewrite
trajectory data.

## Load data

In the browser, choose or type the server directory containing:

```text
all_trajectories.jsonl
```

For example:

```text
/data/zhangdw12/outputs/alfworld-teacher
```

Click **Load**. The Node service streams the JSONL file, builds an in-memory
index, and keeps byte offsets so full trajectory details can be loaded on demand.

## What it shows

- Overview cards: total trajectories, unique samples, success rate, success@any,
  complete-sample success@8, and SFT candidate count.
- Status distribution, token/step averages, invalid-action summaries, and task
  type success rates.
- Overview page (`/`): summary cards, task/status analysis, paginated sample,
  trajectory, and SFT-candidate tables.
- Sample page (`/sample.html?id=<sample_id>`): one sample's attempts in a
  focused page, plus an action transition graph. The graph merges the same
  action into one shared node even when sampled trajectories diverge and later
  converge, with edges showing observed adjacent-action transitions. The
  default graph view is compact: it shows high-frequency transitions first,
  supports an all-transition toggle, and lets you click a node to focus all of
  its incoming/outgoing transitions.
- Trajectory page (`/trajectory.html?id=<trajectory_id>`): step-by-step
  observation, system/user prompt, teacher response, admissible actions, parsed
  action, validity, reward, and token counts.

## Development checks

```bash
npm test
npm run check
```
