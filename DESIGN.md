# Design

## Source of truth
- Status: Draft
- Last refreshed: 2026-06-30
- Primary product surfaces: `examples/alfworld/trajectory_viewer` web UI for inspecting ALFWorld teacher trajectories.
- Evidence reviewed:
  - `examples/alfworld/collect_teacher_trajectories.py`: writes a single append-only `all_trajectories.jsonl` ledger with status, terminal reason, step records, token counts, and resume IDs.
  - `examples/alfworld/build_sft_from_teacher_trajectories.py`: selects the shortest successful trajectory per sample for SFT construction.
  - `examples/alfworld/README.md`: documents the teacher collection and SFT workflow.
  - Repo inspection found no existing npm/frontend app or design system.

## Brand
- Personality: engineering-first, dense but readable, inspection-oriented.
- Trust signals: exact file paths, row counts, status counts, parse-error visibility, and read-only wording.
- Avoid: marketing dashboards, hidden filtering, or UI that implies it mutates training data.

## Product goals
- Goals:
  - Let a researcher load a server directory containing `all_trajectories.jsonl` from a browser.
  - Show all teacher sampling outcomes for analysis, not only successful samples.
  - Make SFT candidate selection auditable from the same raw ledger.
- Non-goals:
  - Editing trajectory JSONL files.
  - Launching teacher sampling jobs.
  - Replacing the Python SFT builder.
- Success signals:
  - `npm run dev` starts a usable local service.
  - Large JSONL files are streamed for indexing instead of loaded by the browser.
  - Users can move from aggregate metrics to sample groups to exact step details.

## Personas and jobs
- Primary personas: researchers and engineers running ALFWorld teacher sampling on a server.
- User jobs:
  - Check whether teacher collection is healthy while or after it runs.
  - Compare successes, failures, truncations, aborts, and errors.
  - Inspect why a specific trajectory failed.
  - Preview which successful trajectories the SFT builder will use.
- Key contexts of use: local browser with SSH port forwarding or a trusted server-local browser session.

## Information architecture
- Primary navigation: single-page dashboard with sections rather than multi-route navigation.
- Core routes/screens:
  - Directory loader/browser.
  - Overview metrics.
  - Sample grouping table.
  - All trajectory table.
  - SFT candidate table.
  - Trajectory detail panel.
- Content hierarchy: dataset identity and health first, aggregate analysis second, row-level inspection third.

## Design principles
- Principle 1: Preserve the raw-ledger mental model. The UI reads `all_trajectories.jsonl` as the only source of truth.
- Principle 2: Keep analysis drill-down reversible. Every aggregate should connect to samples or trajectory IDs.
- Tradeoffs: prefer a zero-dependency implementation and server-side indexing over richer chart libraries or complex build tooling.

## Visual language
- Color: dark analytical UI with high-contrast text; status badges use green for success, red for failure/error, amber for truncation.
- Typography: system sans for UI, monospace for IDs and paths.
- Spacing/layout rhythm: card-and-panel layout with dense tables and scrollable detail blocks.
- Shape/radius/elevation: rounded panels and subtle borders for section separation.
- Motion: none required.
- Imagery/iconography: none required.

## Components
- Existing components to reuse: none found.
- New/changed components:
  - Directory loader and server-side directory browser.
  - Metric cards and CSS bar chart.
  - Filtered/paginated samples, trajectories, and SFT candidate tables.
  - Step detail cards.
- Variants and states: loading, empty, API error, no dataset, loaded dataset, invalid trajectory row.
- Token/component ownership: local CSS in `examples/alfworld/trajectory_viewer/public/styles.css` only.

## Accessibility
- Target standard: practical keyboard-readable local research tool.
- Keyboard/focus behavior: native buttons, inputs, and selects remain focusable.
- Contrast/readability: dark background with high-contrast text and visible table borders.
- Screen-reader semantics: headings, tables, buttons, and labels use native HTML semantics.
- Reduced motion and sensory considerations: no animation-dependent functionality.

## Responsive behavior
- Supported breakpoints/devices: desktop-first; usable on narrower laptop screens.
- Layout adaptations: metric cards and table filters collapse at smaller widths; tables remain horizontally scrollable.
- Touch/hover differences: click targets remain native buttons.

## Interaction states
- Loading: message panel reports indexing progress.
- Empty: panels show no-data messages before load.
- Error: API errors appear in the message panel.
- Success: loaded row count and path are shown after indexing.
- Disabled: load controls are disabled while indexing.
- Offline/slow network, if applicable: intended to run locally; large-file indexing may take time but does not depend on external network.

## Content voice
- Tone: concise, operational, evidence-first.
- Terminology: use `trajectory`, `sample`, `attempt`, `success`, `failed`, `truncated`, `aborted`, `error`, and `SFT candidate` consistently.
- Microcopy rules: call out that paths are server-side paths and the viewer is read-only.

## Implementation constraints
- Framework/styling system: zero-dependency Node HTTP server plus native HTML/CSS/JS; no Vite/React dependency required.
- Design-token constraints: CSS variables local to the viewer.
- Performance constraints: stream JSONL server-side; store summaries and byte offsets, then load full detail rows on demand.
- Compatibility constraints: Node.js >= 20; tested with Node 23.11.0.
- Test/screenshot expectations: Node unit tests for parsing/analysis; manual browser smoke or HTTP smoke for UI.

## Open questions
- [ ] Whether future versions should support live tailing of a growing `all_trajectories.jsonl` during active collection / owner: ALFWorld experiment owner / impact: monitoring UX.
- [ ] Whether analysis should add exportable CSV/Parquet summaries / owner: downstream analysis user / impact: workflow integration.
