import assert from 'node:assert/strict';
import test from 'node:test';

import { actionForStep, buildActionGraph, selectActionGraphView, summarizeActionGraph } from './tree.js';

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
