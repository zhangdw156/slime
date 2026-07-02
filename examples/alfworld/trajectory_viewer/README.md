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
  converge. The default graph view overlays one colored path per sampled
  trajectory on the shared action nodes, includes a trajectory legend, and lets
  you click a trajectory to highlight its full path. The selected trajectory's
  order is shown on node badges (edges do not carry step numbers), and the path
  strip keeps repeated visits to the same merged action readable. Nodes stay
  compact by default; clicking a node opens a detail panel
  with visit counts, status mix, trajectory ids, and an optional button to focus
  its incoming/outgoing transitions. The same page also renders a state
  transition graph whose nodes are observations and whose edges are actions, so
  branch points show the concrete observation states that led to different next
  actions. A major-transition toggle is still available for compact aggregate
  views.
- Trajectory page (`/trajectory.html?id=<trajectory_id>`): step-by-step
  observation, system/user prompt, teacher response, admissible actions, parsed
  action, validity, reward, and token counts.

## Development checks

```bash
npm test
npm run check
```
