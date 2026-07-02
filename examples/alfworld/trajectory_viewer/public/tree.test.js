import assert from 'node:assert/strict';
import test from 'node:test';

import {
  actionForStep,
  buildActionGraph,
  buildStateGraph,
  nodeOrderEntriesForPath,
  nodeOrderPhaseMapForPath,
  selectActionGraphView,
  summarizeActionGraph,
  summarizeStateGraph,
  trajectorySegmentsForView,
} from './tree.js';

test('actionForStep falls back across known action fields', () => {
  assert.equal(actionForStep({ action: 'open fridge' }), 'open fridge');
  assert.equal(actionForStep({ parsed_action: 'take apple' }), 'take apple');
  assert.equal(actionForStep({ teacher_action: 'look' }), 'look');
  assert.equal(actionForStep({}), '(no action recorded)');
});

test('buildActionGraph merges the same action after divergent prefixes', () => {
  const rows = [
    {
      trajectory_id: 'a',
      status: 'success',
      steps: [{ action: 'look' }, { action: 'open fridge' }, { action: 'take apple' }],
    },
    {
      trajectory_id: 'b',
      status: 'failed',
      steps: [{ action: 'look' }, { action: 'open cabinet' }, { action: 'take apple' }],
    },
    {
      trajectory_id: 'c',
      status: 'failed',
      steps: [{ action: 'look' }, { action: 'open cabinet' }, { action: 'look' }],
    },
  ];

  const graph = buildActionGraph(rows);
  const summary = summarizeActionGraph(graph);
  const actionNodes = graph.nodes.filter((node) => node.kind === 'action');
  const look = graph.nodes.find((node) => node.id === 'action:look');
  const takeApple = graph.nodes.find((node) => node.id === 'action:take apple');

  assert.equal(graph.trajectories, 3);
  assert.equal(actionNodes.length, 4);
  assert.equal(look.attempts, 3);
  assert.equal(look.occurrenceCount, 4);
  assert.equal(takeApple.attempts, 2);
  assert.equal(takeApple.incoming.length, 2);
  assert.equal(graph.edges.find((edge) => edge.from === 'action:open fridge' && edge.to === 'action:take apple').count, 1);
  assert.equal(graph.edges.find((edge) => edge.from === 'action:open cabinet' && edge.to === 'action:take apple').count, 1);
  assert.equal(summary.convergencePoints >= 1, true);
  assert.equal(summary.maxPathLength, 3);
});

test('buildStateGraph uses observations as nodes and actions as branching edges', () => {
  const rows = [
    {
      trajectory_id: 'a',
      status: 'success',
      steps: [
        { observation: 'You are in a kitchen.', action: 'open fridge' },
        { observation: 'The fridge is open.', action: 'take apple' },
      ],
    },
    {
      trajectory_id: 'b',
      status: 'failed',
      steps: [
        { observation: 'You are in a kitchen.', action: 'open cabinet' },
        { observation: 'The cabinet is open.', action: 'look' },
      ],
    },
  ];

  const graph = buildStateGraph(rows);
  const summary = summarizeStateGraph(graph);
  const kitchen = graph.nodes.find((node) => node.description === 'You are in a kitchen.');
  const outgoing = kitchen.outgoing.map((id) => graph.edges.find((edge) => edge.id === id).label).sort();

  assert.equal(summary.uniqueStates, 3);
  assert.equal(summary.actionEdges, 4);
  assert.equal(summary.branchPoints, 1);
  assert.deepEqual(outgoing, ['open cabinet', 'open fridge']);
  assert.equal(graph.edges.every((edge) => edge.label), true);
  assert.deepEqual(graph.trajectoryPaths[0].nodeIds.map((id) => graph.nodes.find((node) => node.id === id)?.description || id), [
    'You are in a kitchen.',
    'The fridge is open.',
    'end:success',
  ]);
});

test('buildActionGraph keeps one ordered path per trajectory while sharing action nodes', () => {
  const rows = [
    {
      trajectory_id: 'sample_00',
      sample_repeat_id: 0,
      status: 'success',
      steps: [{ action: 'look' }, { action: 'open fridge' }, { action: 'take apple' }],
    },
    {
      trajectory_id: 'sample_01',
      sample_repeat_id: 1,
      status: 'failed',
      steps: [{ action: 'look' }, { action: 'open cabinet' }, { action: 'take apple' }],
    },
  ];

  const graph = buildActionGraph(rows);
  const lookNodes = graph.nodes.filter((node) => node.id === 'action:look');

  assert.equal(lookNodes.length, 1);
  assert.equal(graph.trajectoryPaths.length, 2);
  assert.deepEqual(graph.trajectoryPaths.map((path) => path.id), ['sample_00', 'sample_01']);
  assert.deepEqual(graph.trajectoryPaths.map((path) => path.repeatId), [0, 1]);
  assert.deepEqual(graph.trajectoryPaths[0].nodeIds, ['start', 'action:look', 'action:open fridge', 'action:take apple', 'end:success']);
  assert.deepEqual(graph.trajectoryPaths[1].nodeIds, ['start', 'action:look', 'action:open cabinet', 'action:take apple', 'end:failed']);
  assert.deepEqual(graph.trajectoryPaths[0].edgeIds, [
    'start→action:look',
    'action:look→action:open fridge',
    'action:open fridge→action:take apple',
    'action:take apple→end:success',
  ]);
});

test('selectActionGraphView exposes visible trajectory paths for the chosen graph scope', () => {
  const rows = [
    { trajectory_id: 'a', status: 'success', steps: [{ action: 'look' }, { action: 'open fridge' }, { action: 'take apple' }] },
    { trajectory_id: 'b', status: 'failed', steps: [{ action: 'look' }, { action: 'open cabinet' }, { action: 'take apple' }] },
  ];
  const graph = buildActionGraph(rows);
  const all = selectActionGraphView(graph, { mode: 'all' });
  const focused = selectActionGraphView(graph, { focusedNodeId: 'action:open fridge' });

  assert.equal(all.paths.length, 2);
  assert.equal(all.paths.every((path) => path.isComplete), true);
  assert.deepEqual(all.paths[0].visibleEdgeIds, graph.trajectoryPaths[0].edgeIds);
  assert.deepEqual(all.paths[1].visibleEdgeIds, graph.trajectoryPaths[1].edgeIds);

  assert.deepEqual(focused.paths.map((path) => [path.id, path.visibleEdgeIds.length]), [
    ['a', 2],
    ['b', 0],
  ]);
  assert.equal(focused.paths[0].isComplete, false);
});

test('trajectorySegmentsForView expands shared transitions into per-trajectory lanes', () => {
  const rows = [
    { trajectory_id: 'a', status: 'success', steps: [{ action: 'look' }, { action: 'open fridge' }] },
    { trajectory_id: 'b', status: 'failed', steps: [{ action: 'look' }, { action: 'open cabinet' }] },
  ];
  const graph = buildActionGraph(rows);
  const view = selectActionGraphView(graph, { mode: 'all' });
  const segments = trajectorySegmentsForView(view);
  const sharedStart = segments.filter((segment) => segment.edgeId === 'start→action:look');

  assert.equal(segments.length, 6);
  assert.deepEqual(sharedStart.map((segment) => segment.trajectoryId), ['a', 'b']);
  assert.deepEqual(sharedStart.map((segment) => segment.status), ['success', 'failed']);
  assert.equal(sharedStart[0].laneOffset + sharedStart[1].laneOffset, 0);
  assert.equal(sharedStart[0].laneOffset < sharedStart[1].laneOffset, true);
});

test('nodeOrderEntriesForPath labels trajectory order on shared action nodes', () => {
  const graph = buildActionGraph([
    {
      trajectory_id: 'repeat-look',
      status: 'failed',
      steps: [{ action: 'look' }, { action: 'open drawer' }, { action: 'look' }],
    },
  ]);

  const entries = nodeOrderEntriesForPath(graph.trajectoryPaths[0]);
  const labelsByNode = Object.fromEntries(entries.map((entry) => [entry.nodeId, entry.labels]));

  assert.deepEqual(entries.map((entry) => entry.nodeId), ['start', 'action:look', 'action:open drawer', 'end:failed']);
  assert.deepEqual(labelsByNode.start, ['S']);
  assert.deepEqual(labelsByNode['action:look'], ['1', '3']);
  assert.deepEqual(labelsByNode['action:open drawer'], ['2']);
  assert.deepEqual(labelsByNode['end:failed'], ['E']);
});

test('nodeOrderPhaseMapForPath orders highlighted trajectory nodes by selected path steps', () => {
  const graph = buildActionGraph([
    {
      trajectory_id: 'chosen',
      status: 'success',
      steps: [{ action: 'go to diningtable 1' }, { action: 'take spatula 1 from diningtable 1' }],
    },
    {
      trajectory_id: 'other',
      status: 'failed',
      steps: [{ action: 'take spatula 1 from diningtable 1' }, { action: 'open drawer 1' }, { action: 'go to diningtable 1' }],
    },
  ]);
  const phases = nodeOrderPhaseMapForPath(graph.trajectoryPaths[0]);

  assert.equal(phases.get('start'), 0);
  assert.equal(phases.get('action:go to diningtable 1'), 1);
  assert.equal(phases.get('action:take spatula 1 from diningtable 1'), 2);
  assert.equal(phases.get('action:go to diningtable 1') < phases.get('action:take spatula 1 from diningtable 1'), true);
  assert.equal(phases.get('end:success'), 3);
});

test('trajectorySegmentsForView keeps original step indices when a focused view hides earlier edges', () => {
  const graph = buildActionGraph([
    { trajectory_id: 'a', status: 'success', steps: [{ action: 'look' }, { action: 'open fridge' }, { action: 'take apple' }] },
  ]);
  const focused = selectActionGraphView(graph, { focusedNodeId: 'action:open fridge' });
  const segments = trajectorySegmentsForView(focused).filter((segment) => segment.trajectoryId === 'a');

  assert.deepEqual(segments.map((segment) => segment.edgeId), ['action:look→action:open fridge', 'action:open fridge→action:take apple']);
  assert.deepEqual(segments.map((segment) => segment.pathStepIndex), [1, 2]);
});

test('selectActionGraphView keeps the default view compact and supports focused adjacency', () => {
  const rows = [
    { trajectory_id: 'a', status: 'success', steps: [{ action: 'look' }, { action: 'open fridge' }, { action: 'take apple' }] },
    { trajectory_id: 'b', status: 'failed', steps: [{ action: 'look' }, { action: 'open cabinet' }, { action: 'take apple' }] },
    { trajectory_id: 'c', status: 'failed', steps: [{ action: 'look' }, { action: 'open cabinet' }, { action: 'look' }] },
  ];
  const graph = buildActionGraph(rows);
  const overview = selectActionGraphView(graph);
  const all = selectActionGraphView(graph, { mode: 'all' });
  const focused = selectActionGraphView(graph, { focusedNodeId: 'action:take apple' });

  assert.equal(overview.mode, 'overview');
  assert.equal(overview.minMainEdgeCount, 2);
  assert.equal(overview.edges.length < graph.edges.length, true);
  assert.equal(overview.hiddenEdges > 0, true);
  assert.equal(all.edges.length, graph.edges.length);
  assert.equal(all.nodes.length, graph.nodes.length);
  assert.equal(focused.mode, 'focus');
  assert.deepEqual(new Set(focused.edges.map((edge) => edge.id)), new Set(graph.edges.filter((edge) => edge.from === 'action:take apple' || edge.to === 'action:take apple').map((edge) => edge.id)));
});
