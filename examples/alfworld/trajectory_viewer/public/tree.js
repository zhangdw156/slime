function stableText(value, fallback = '') {
  if (value === null || value === undefined) return fallback;
  const text = typeof value === 'object' ? stableJson(value) : String(value).trim();
  return text || fallback;
}

function stableJson(value) {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value).trim();
  }
}

function statusForRow(row) {
  if (row?.status) return String(row.status);
  if (row?.error) return 'error';
  if (row?.aborted) return 'aborted';
  if (row?.reward === 1 || row?.won === true) return 'success';
  if (row?.truncated) return 'truncated';
  if (row?.done) return 'failed';
  return 'unknown';
}

function trajectoryIdForRow(row, index) {
  return stableText(row?.trajectory_id ?? row?.episode_id, `trajectory-${index + 1}`);
}

function repeatIdForRow(row) {
  return row?.sample_repeat_id ?? row?.attempt_id ?? null;
}

function sampleIdForRow(row) {
  return row?.sample_id ?? row?.task_index ?? row?.gamefile ?? null;
}

function terminalReasonForRow(row) {
  return stableText(row?.terminal_reason || row?.error || row?.finish_reason || 'terminal');
}

function addToSetMap(map, key, value) {
  if (!map.has(key)) map.set(key, new Set());
  map.get(key).add(value);
}

function addStatusSet(statusSets, status, trajectoryId) {
  addToSetMap(statusSets, status, trajectoryId);
}

function statusCountsFromSets(statusSets) {
  return Object.fromEntries([...statusSets.entries()].map(([status, values]) => [status, values.size]).sort(([a], [b]) => a.localeCompare(b)));
}

function hashText(value) {
  const text = String(value || '');
  let hash = 5381;
  for (let index = 0; index < text.length; index += 1) {
    hash = ((hash << 5) + hash) ^ text.charCodeAt(index);
  }
  return (hash >>> 0).toString(36);
}

function compactLabel(value, fallback = 'unknown', maxLength = 80) {
  const text = stableText(value, fallback).replace(/\s+/g, ' ');
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1)}…`;
}

function makeNode(id, label, kind = 'action', description = '') {
  return {
    id,
    label,
    kind,
    description,
    trajectoryIds: new Set(),
    statusSets: new Map(),
    occurrenceCount: 0,
    stepTotal: 0,
    firstStep: Infinity,
    terminalReasons: new Map(),
    incoming: new Set(),
    outgoing: new Set(),
  };
}

function makeEdge(id, from, to, label = '') {
  return {
    id,
    from,
    to,
    label,
    count: 0,
    trajectoryIds: new Set(),
    statusSets: new Map(),
  };
}

function edgeId(from, to, label = '') {
  return label ? `${from}→${to}#${hashText(label)}` : `${from}→${to}`;
}

function actionId(action) {
  return `action:${action}`;
}

function stateId(observation) {
  const text = stableText(observation, '(no observation recorded)');
  return `state:${hashText(text)}:${text.length}`;
}

function endId(status) {
  return `end:${status}`;
}

function registerNodeVisit(node, trajectoryId, status, stepIndex = null) {
  node.trajectoryIds.add(trajectoryId);
  addStatusSet(node.statusSets, status, trajectoryId);
  if (stepIndex !== null) {
    node.occurrenceCount += 1;
    node.stepTotal += stepIndex;
    node.firstStep = Math.min(node.firstStep, stepIndex);
  }
}

function registerEdge(edges, nodes, from, to, trajectoryId, status, options = {}) {
  const label = stableText(options.label, '');
  const id = edgeId(from, to, label);
  if (!edges.has(id)) edges.set(id, makeEdge(id, from, to, label));
  const edge = edges.get(id);
  edge.count += 1;
  edge.trajectoryIds.add(trajectoryId);
  addStatusSet(edge.statusSets, status, trajectoryId);
  nodes.get(from)?.outgoing.add(id);
  nodes.get(to)?.incoming.add(id);
  return edge;
}

function makeTrajectoryPath(row, index, status, nodeIds, edgeIds) {
  const trajectoryId = trajectoryIdForRow(row, index);
  const steps = Array.isArray(row?.steps) ? row.steps : [];
  return {
    id: trajectoryId,
    index,
    repeatId: repeatIdForRow(row),
    sampleId: sampleIdForRow(row),
    status,
    terminalReason: terminalReasonForRow(row),
    stepCount: steps.length,
    nodeIds,
    edgeIds,
  };
}

function finalizeNode(node, maxPathLength) {
  const trajectoryIds = [...node.trajectoryIds].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const avgStep = node.occurrenceCount ? node.stepTotal / node.occurrenceCount : 0;
  let phase = 0;
  if (node.kind === 'action' || node.kind === 'state') {
    const phaseSize = Math.max(1, Math.ceil(Math.max(maxPathLength, 1) / 8));
    phase = 1 + Math.floor(Math.max(avgStep - 1, 0) / phaseSize);
  } else if (node.kind === 'end') {
    phase = 9;
  }

  return {
    id: node.id,
    label: node.label,
    kind: node.kind,
    description: node.description,
    trajectoryIds,
    attempts: trajectoryIds.length,
    occurrenceCount: node.occurrenceCount,
    avgStep,
    firstStep: Number.isFinite(node.firstStep) ? node.firstStep : null,
    phase,
    statusCounts: statusCountsFromSets(node.statusSets),
    terminalReasons: Object.fromEntries([...node.terminalReasons.entries()].sort(([a], [b]) => a.localeCompare(b))),
    incoming: [...node.incoming].sort(),
    outgoing: [...node.outgoing].sort(),
  };
}

function finalizeEdge(edge) {
  return {
    id: edge.id,
    from: edge.from,
    to: edge.to,
    label: edge.label,
    count: edge.count,
    attempts: edge.trajectoryIds.size,
    trajectoryIds: [...edge.trajectoryIds].sort((a, b) => a.localeCompare(b, undefined, { numeric: true })),
    statusCounts: statusCountsFromSets(edge.statusSets),
  };
}

export function actionForStep(step) {
  return stableText(step?.action ?? step?.parsed_action ?? step?.teacher_action, '(no action recorded)');
}

export function observationForStep(step) {
  return stableText(step?.observation ?? step?.obs ?? step?.state ?? step?.text_observation, '(no observation recorded)');
}

export function buildActionGraph(rows = []) {
  const nodes = new Map();
  const edges = new Map();
  const trajectoryPaths = [];
  const start = makeNode('start', 'START', 'start');
  nodes.set(start.id, start);
  let maxPathLength = 0;

  rows.forEach((row, index) => {
    const trajectoryId = trajectoryIdForRow(row, index);
    const status = statusForRow(row);
    const steps = Array.isArray(row?.steps) ? row.steps : [];
    maxPathLength = Math.max(maxPathLength, steps.length);
    let previousId = start.id;
    const nodeIds = [start.id];
    const edgeIds = [];

    registerNodeVisit(start, trajectoryId, status);

    steps.forEach((step, stepIndex) => {
      const action = actionForStep(step);
      const currentId = actionId(action);
      if (!nodes.has(currentId)) nodes.set(currentId, makeNode(currentId, action, 'action'));
      const current = nodes.get(currentId);
      registerNodeVisit(current, trajectoryId, status, stepIndex + 1);
      registerEdge(edges, nodes, previousId, currentId, trajectoryId, status);
      edgeIds.push(edgeId(previousId, currentId));
      nodeIds.push(currentId);
      previousId = currentId;
    });

    const terminalStatus = status || 'unknown';
    const currentEndId = endId(terminalStatus);
    if (!nodes.has(currentEndId)) nodes.set(currentEndId, makeNode(currentEndId, `END · ${terminalStatus}`, 'end'));
    const end = nodes.get(currentEndId);
    registerNodeVisit(end, trajectoryId, status);
    const terminalReason = terminalReasonForRow(row);
    end.terminalReasons.set(terminalReason, (end.terminalReasons.get(terminalReason) || 0) + 1);
    registerEdge(edges, nodes, previousId, currentEndId, trajectoryId, status);
    edgeIds.push(edgeId(previousId, currentEndId));
    nodeIds.push(currentEndId);
    trajectoryPaths.push(makeTrajectoryPath(row, index, status, nodeIds, edgeIds));
  });

  const nodeList = [...nodes.values()].map((node) => finalizeNode(node, maxPathLength));
  const edgeList = [...edges.values()].map(finalizeEdge);
  nodeList.sort((a, b) => a.phase - b.phase || b.attempts - a.attempts || a.label.localeCompare(b.label));
  edgeList.sort((a, b) => b.count - a.count || a.from.localeCompare(b.from) || a.to.localeCompare(b.to));

  return {
    nodes: nodeList,
    edges: edgeList,
    trajectoryPaths,
    maxPathLength,
    trajectories: rows.length,
  };
}

export function buildStateGraph(rows = []) {
  const nodes = new Map();
  const edges = new Map();
  const trajectoryPaths = [];
  let maxPathLength = 0;

  rows.forEach((row, index) => {
    const trajectoryId = trajectoryIdForRow(row, index);
    const status = statusForRow(row);
    const steps = Array.isArray(row?.steps) ? row.steps : [];
    maxPathLength = Math.max(maxPathLength, steps.length);
    const nodeIds = [];
    const edgeIds = [];

    let previousId = '';
    steps.forEach((step, stepIndex) => {
      const observation = observationForStep(step);
      const currentId = stateId(observation);
      if (!nodes.has(currentId)) {
        nodes.set(currentId, makeNode(currentId, compactLabel(observation, '(no observation recorded)', 72), 'state', observation));
      }
      registerNodeVisit(nodes.get(currentId), trajectoryId, status, stepIndex + 1);
      nodeIds.push(currentId);

      if (previousId) {
        const previousAction = actionForStep(steps[stepIndex - 1]);
        const edge = registerEdge(edges, nodes, previousId, currentId, trajectoryId, status, { label: previousAction });
        edgeIds.push(edge.id);
      }
      previousId = currentId;
    });

    const terminalStatus = status || 'unknown';
    const currentEndId = endId(terminalStatus);
    if (!nodes.has(currentEndId)) nodes.set(currentEndId, makeNode(currentEndId, `END · ${terminalStatus}`, 'end'));
    const end = nodes.get(currentEndId);
    registerNodeVisit(end, trajectoryId, status);
    const terminalReason = terminalReasonForRow(row);
    end.terminalReasons.set(terminalReason, (end.terminalReasons.get(terminalReason) || 0) + 1);

    if (previousId) {
      const finalAction = actionForStep(steps[steps.length - 1]);
      const edge = registerEdge(edges, nodes, previousId, currentEndId, trajectoryId, status, { label: finalAction });
      edgeIds.push(edge.id);
    } else {
      const startId = 'state:no-observation';
      if (!nodes.has(startId)) nodes.set(startId, makeNode(startId, 'NO OBSERVATION', 'state', '(no observation recorded)'));
      registerNodeVisit(nodes.get(startId), trajectoryId, status);
      nodeIds.push(startId);
      const edge = registerEdge(edges, nodes, startId, currentEndId, trajectoryId, status, { label: '(no action recorded)' });
      edgeIds.push(edge.id);
    }
    nodeIds.push(currentEndId);
    trajectoryPaths.push(makeTrajectoryPath(row, index, status, nodeIds, edgeIds));
  });

  const nodeList = [...nodes.values()].map((node) => finalizeNode(node, maxPathLength));
  const edgeList = [...edges.values()].map(finalizeEdge);
  nodeList.sort((a, b) => a.phase - b.phase || b.attempts - a.attempts || a.label.localeCompare(b.label));
  edgeList.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label) || a.from.localeCompare(b.from) || a.to.localeCompare(b.to));

  return {
    nodes: nodeList,
    edges: edgeList,
    trajectoryPaths,
    maxPathLength,
    trajectories: rows.length,
  };
}

export function statusEntries(statusCounts = {}) {
  return Object.entries(statusCounts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

export function nodeOrderEntriesForPath(path, options = {}) {
  const visibleNodeIds = options.visibleNodeIds ? new Set(options.visibleNodeIds) : null;
  const entries = new Map();
  const nodeIds = path?.nodeIds || [];
  const finalIndex = Math.max(nodeIds.length - 1, 0);

  nodeIds.forEach((nodeId, index) => {
    if (visibleNodeIds && !visibleNodeIds.has(nodeId)) return;
    const label = index === 0 ? 'S' : index === finalIndex ? 'E' : String(index);
    if (!entries.has(nodeId)) entries.set(nodeId, { nodeId, labels: [], positions: [] });
    const entry = entries.get(nodeId);
    entry.labels.push(label);
    entry.positions.push(index);
  });

  return [...entries.values()];
}

export function nodeOrderPhaseMapForPath(path, options = {}) {
  return new Map(nodeOrderEntriesForPath(path, options).map((entry) => [entry.nodeId, Math.min(...entry.positions)]));
}

export function trajectorySegmentsForView(view, options = {}) {
  const laneStep = Number.isFinite(options.laneStep) ? options.laneStep : 7;
  const edgeById = new Map((view?.edges || []).map((edge) => [edge.id, edge]));
  const rawSegments = [];

  for (const path of view?.paths || []) {
    const visibleEdgeIds = new Set(path.visibleEdgeIds || path.edgeIds || []);
    (path.edgeIds || []).forEach((edgeId, pathStepIndex) => {
      if (!visibleEdgeIds.has(edgeId)) return;
      const edge = edgeById.get(edgeId);
      if (!edge) return;
      rawSegments.push({
        id: `${path.id}:${pathStepIndex}:${edgeId}`,
        edge,
        edgeId,
        from: edge.from,
        to: edge.to,
        trajectoryId: path.id,
        trajectoryIndex: path.index,
        repeatId: path.repeatId,
        status: path.status,
        stepCount: path.stepCount,
        pathStepIndex,
        pathComplete: path.isComplete,
      });
    });
  }

  const byEdge = new Map();
  for (const segment of rawSegments) addToSetMap(byEdge, segment.edgeId, segment);

  const laneOffsets = new Map();
  for (const [edgeId, segmentSet] of byEdge.entries()) {
    const segments = [...segmentSet].sort(
      (a, b) => a.trajectoryIndex - b.trajectoryIndex || a.pathStepIndex - b.pathStepIndex || a.trajectoryId.localeCompare(b.trajectoryId, undefined, { numeric: true }),
    );
    segments.forEach((segment, laneIndex) => {
      laneOffsets.set(segment.id, {
        laneIndex,
        laneCount: segments.length,
        laneOffset: (laneIndex - (segments.length - 1) / 2) * laneStep,
      });
    });
    byEdge.set(edgeId, segments);
  }

  return rawSegments.map((segment) => ({
    ...segment,
    ...laneOffsets.get(segment.id),
  }));
}

export function summarizeActionGraph(graph) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const actionNodes = nodes.filter((node) => node.kind === 'action');
  return {
    trajectories: graph?.trajectories || 0,
    uniqueActions: actionNodes.length,
    transitionEdges: edges.length,
    branchPoints: actionNodes.filter((node) => node.outgoing.length > 1).length,
    convergencePoints: actionNodes.filter((node) => node.incoming.length > 1).length,
    maxPathLength: graph?.maxPathLength || 0,
  };
}

export function summarizeStateGraph(graph) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const stateNodes = nodes.filter((node) => node.kind === 'state');
  return {
    trajectories: graph?.trajectories || 0,
    uniqueStates: stateNodes.length,
    actionEdges: edges.length,
    branchPoints: stateNodes.filter((node) => node.outgoing.length > 1).length,
    convergencePoints: stateNodes.filter((node) => node.incoming.length > 1).length,
    maxPathLength: graph?.maxPathLength || 0,
  };
}

function sortEdgesByWeight(a, b) {
  return b.count - a.count || b.attempts - a.attempts || a.from.localeCompare(b.from) || a.to.localeCompare(b.to);
}

function addEdge(selection, edge) {
  if (!edge) return;
  selection.set(edge.id, edge);
}

function nodesForEdges(nodes, edges, extraNodeIds = []) {
  const ids = new Set(extraNodeIds);
  for (const edge of edges) {
    ids.add(edge.from);
    ids.add(edge.to);
  }
  return nodes.filter((node) => ids.has(node.id));
}

function trajectoryPathsForView(graph, visibleNodes, visibleEdges) {
  const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
  const visibleEdgeIds = new Set(visibleEdges.map((edge) => edge.id));
  return (graph?.trajectoryPaths || []).map((path) => {
    const pathVisibleEdgeIds = path.edgeIds.filter((id) => visibleEdgeIds.has(id));
    return {
      ...path,
      visibleNodeIds: path.nodeIds.filter((id) => visibleNodeIds.has(id)),
      visibleEdgeIds: pathVisibleEdgeIds,
      isComplete: pathVisibleEdgeIds.length === path.edgeIds.length,
    };
  });
}

export function selectActionGraphView(graph, options = {}) {
  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  const nodeById = new Map(nodes.map((node) => [node.id, node]));
  const requestedMode = options.mode === 'all' ? 'all' : 'overview';
  const focusedNodeId = options.focusedNodeId && nodeById.has(options.focusedNodeId) ? options.focusedNodeId : '';

  if (!nodes.length) {
    return {
      mode: requestedMode,
      focusedNodeId,
      nodes: [],
      edges: [],
      paths: [],
      hiddenNodes: 0,
      hiddenEdges: 0,
      minMainEdgeCount: 1,
      isComplete: true,
    };
  }

  if (focusedNodeId) {
    const visibleEdges = edges.filter((edge) => edge.from === focusedNodeId || edge.to === focusedNodeId).sort(sortEdgesByWeight);
    const visibleNodes = nodesForEdges(nodes, visibleEdges, [focusedNodeId]);
    return {
      mode: 'focus',
      focusedNodeId,
      nodes: visibleNodes,
      edges: visibleEdges,
      paths: trajectoryPathsForView(graph, visibleNodes, visibleEdges),
      hiddenNodes: Math.max(nodes.length - visibleNodes.length, 0),
      hiddenEdges: Math.max(edges.length - visibleEdges.length, 0),
      minMainEdgeCount: 1,
      isComplete: visibleEdges.length === edges.length && visibleNodes.length === nodes.length,
    };
  }

  if (requestedMode === 'all') {
    return {
      mode: 'all',
      focusedNodeId: '',
      nodes,
      edges,
      paths: trajectoryPathsForView(graph, nodes, edges),
      hiddenNodes: 0,
      hiddenEdges: 0,
      minMainEdgeCount: 1,
      isComplete: true,
    };
  }

  const minMainEdgeCount = Math.max(1, Math.ceil((graph?.trajectories || 1) * 0.5));
  const maxOverviewEdges = Math.max(16, (graph?.trajectories || 1) * 3);
  const selected = new Map();
  const mainEdges = edges.filter((edge) => edge.count >= minMainEdgeCount).sort(sortEdgesByWeight);
  for (const edge of mainEdges.slice(0, maxOverviewEdges)) addEdge(selected, edge);

  const startEdges = edges.filter((edge) => edge.from === 'start').sort(sortEdgesByWeight);
  for (const edge of startEdges.slice(0, 3)) addEdge(selected, edge);

  const selectedNodeIds = new Set();
  for (const edge of selected.values()) {
    selectedNodeIds.add(edge.from);
    selectedNodeIds.add(edge.to);
  }

  const terminalEdges = edges
    .filter((edge) => edge.to.startsWith('end:') && (selectedNodeIds.has(edge.from) || edge.count >= minMainEdgeCount))
    .sort(sortEdgesByWeight);
  for (const edge of terminalEdges.slice(0, 3)) addEdge(selected, edge);

  if (!selected.size) {
    for (const edge of [...edges].sort(sortEdgesByWeight).slice(0, maxOverviewEdges)) addEdge(selected, edge);
  }

  const visibleEdges = [...selected.values()].sort(sortEdgesByWeight);
  const visibleNodes = nodesForEdges(nodes, visibleEdges, ['start']);
  return {
    mode: 'overview',
    focusedNodeId: '',
    nodes: visibleNodes,
    edges: visibleEdges,
    paths: trajectoryPathsForView(graph, visibleNodes, visibleEdges),
    hiddenNodes: Math.max(nodes.length - visibleNodes.length, 0),
    hiddenEdges: Math.max(edges.length - visibleEdges.length, 0),
    minMainEdgeCount,
    isComplete: visibleEdges.length === edges.length && visibleNodes.length === nodes.length,
  };
}
