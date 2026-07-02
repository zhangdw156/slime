import {
  $,
  api,
  card,
  currentQuery,
  ensureDatasetLoaded,
  escapeHtml,
  formatNumber,
  renderTrajectoryTable,
  setServerStatus,
  showMessage,
  statusBadge,
} from './common.js';
import {
  buildActionGraph,
  buildStateGraph,
  nodeOrderEntriesForPath,
  nodeOrderPhaseMapForPath,
  selectActionGraphView,
  statusEntries,
  summarizeActionGraph,
  summarizeStateGraph,
  trajectorySegmentsForView,
} from './tree.js';

const sampleId = currentQuery().get('id') || '';
let currentActionGraph = null;
let graphViewMode = 'all';
let graphFocusNodeId = '';
let graphHighlightedTrajectoryId = '';
let graphSelectedNodeId = '';
let currentStateGraph = null;
let stateGraphViewMode = 'all';
let stateGraphFocusNodeId = '';

const TRAJECTORY_COLORS = [
  '#38bdf8',
  '#a78bfa',
  '#f97316',
  '#22c55e',
  '#f43f5e',
  '#eab308',
  '#14b8a6',
  '#fb7185',
  '#60a5fa',
  '#c084fc',
  '#34d399',
  '#facc15',
];

function trajectoryColor(index = 0) {
  return TRAJECTORY_COLORS[Math.abs(index) % TRAJECTORY_COLORS.length];
}

function trajectoryLabel(path) {
  return `T${formatNumber((path?.index ?? 0) + 1)}`;
}

function compactTrajectoryId(value) {
  const text = String(value || 'trajectory');
  if (text.length <= 36) return text;
  return `${text.slice(0, 14)}…${text.slice(-16)}`;
}

function compactInline(value, maxLength = 36) {
  const text = String(value || '').replace(/\s+/g, ' ').trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, maxLength - 1)}…`;
}

function renderSample(sample, trajectories) {
  $('sampleSubtitle').textContent = `Sample ${sample.sampleId} · ${sample.taskType} · ${sample.split}`;
  $('sampleCards').innerHTML = [
    card('Sample', sample.sampleId, sample.split),
    card('Task', sample.taskType, sample.split),
    card('Attempts', formatNumber(sample.attempts), `${formatNumber(sample.successes)} successes`),
    card('Status', sample.success ? 'success@any' : 'no success', Object.entries(sample.statusCounts).map(([status, count]) => `${status}:${count}`).join(' ')),
    card('Best success', sample.bestSuccessId || 'none', sample.bestSuccessSteps !== null ? `${formatNumber(sample.bestSuccessSteps)} steps` : ''),
    card('Invalid actions', formatNumber(sample.invalidActionCount), `avg steps ${formatNumber(sample.avgSteps, 2)}`),
  ].join('');
  $('attemptsTable').innerHTML = renderTrajectoryTable(trajectories, { includeSample: false });
}

function renderStatusCounts(statusCounts) {
  return statusEntries(statusCounts)
    .slice(0, 2)
    .map(([status, count]) => `${statusBadge(status)}<span class="count">${formatNumber(count)}</span>`)
    .join('');
}

function edgeGroupOffsets(edges, nodeId, side, positions) {
  const group = edges
    .filter((edge) => (side === 'out' ? edge.from === nodeId : edge.to === nodeId))
    .sort((a, b) => {
      const aOther = positions.get(side === 'out' ? a.to : a.from);
      const bOther = positions.get(side === 'out' ? b.to : b.from);
      return (aOther?.y ?? 0) - (bOther?.y ?? 0) || b.count - a.count || a.id.localeCompare(b.id);
    });
  return group.map((edge, index) => [edge.id, (index - (group.length - 1) / 2) * 5]);
}

function layoutGraph(view, orderPhaseMap = new Map(), options = {}) {
  const nodeWidth = options.nodeWidth || 156;
  const nodeHeight = options.nodeHeight || 48;
  const columnGap = options.columnGap || 128;
  const rowGap = options.rowGap || 20;
  const margin = options.margin || 30;
  const grouped = new Map();
  const nodeById = new Map(view.nodes.map((node) => [node.id, node]));
  const incoming = new Map();
  const outgoing = new Map();

  for (const edge of view.edges) {
    if (!incoming.has(edge.to)) incoming.set(edge.to, []);
    if (!outgoing.has(edge.from)) outgoing.set(edge.from, []);
    incoming.get(edge.to).push(edge);
    outgoing.get(edge.from).push(edge);
  }

  for (const node of view.nodes) {
    const layoutPhase = orderPhaseMap.get(node.id) ?? node.phase;
    if (!grouped.has(layoutPhase)) grouped.set(layoutPhase, []);
    grouped.get(layoutPhase).push(node);
  }

  const phases = [...grouped.keys()].sort((a, b) => a - b);
  const order = new Map();
  const kindOrder = { start: 0, action: 1, state: 1, end: 2 };
  const sortWithinPhase = (nodes) => nodes.sort((a, b) => (kindOrder[a.kind] ?? 1) - (kindOrder[b.kind] ?? 1) || b.attempts - a.attempts || b.occurrenceCount - a.occurrenceCount || a.label.localeCompare(b.label));
  for (const phase of phases) sortWithinPhase(grouped.get(phase));

  function refreshOrder() {
    for (const nodes of grouped.values()) nodes.forEach((node, index) => order.set(node.id, index));
  }

  function neighborAverage(node, direction) {
    const source = direction === 'in' ? incoming.get(node.id) : outgoing.get(node.id);
    if (!source?.length) return null;
    let total = 0;
    let weight = 0;
    for (const edge of source) {
      const neighborId = direction === 'in' ? edge.from : edge.to;
      if (!nodeById.has(neighborId) || !order.has(neighborId)) continue;
      total += order.get(neighborId) * edge.count;
      weight += edge.count;
    }
    return weight ? total / weight : null;
  }

  refreshOrder();
  for (let pass = 0; pass < 4; pass += 1) {
    for (const phase of phases) {
      grouped.get(phase).sort((a, b) => (neighborAverage(a, 'in') ?? order.get(a.id)) - (neighborAverage(b, 'in') ?? order.get(b.id)) || b.attempts - a.attempts || a.label.localeCompare(b.label));
      refreshOrder();
    }
    for (const phase of [...phases].reverse()) {
      grouped.get(phase).sort((a, b) => (neighborAverage(a, 'out') ?? order.get(a.id)) - (neighborAverage(b, 'out') ?? order.get(b.id)) || b.attempts - a.attempts || a.label.localeCompare(b.label));
      refreshOrder();
    }
  }

  const positions = new Map();
  const maxRows = Math.max(...[...grouped.values()].map((nodes) => nodes.length), 1);
  const maxColumnHeight = maxRows * nodeHeight + Math.max(maxRows - 1, 0) * rowGap;
  phases.forEach((phase, columnIndex) => {
    const nodes = grouped.get(phase);
    const columnHeight = nodes.length * nodeHeight + Math.max(nodes.length - 1, 0) * rowGap;
    const yOffset = (maxColumnHeight - columnHeight) / 2;
    nodes.forEach((node, rowIndex) => {
      positions.set(node.id, {
        x: margin + columnIndex * (nodeWidth + columnGap),
        y: margin + yOffset + rowIndex * (nodeHeight + rowGap),
      });
    });
  });

  const sourceOffsets = new Map();
  const targetOffsets = new Map();
  for (const node of view.nodes) {
    for (const [edgeId, offset] of edgeGroupOffsets(view.edges, node.id, 'out', positions)) sourceOffsets.set(edgeId, offset);
    for (const [edgeId, offset] of edgeGroupOffsets(view.edges, node.id, 'in', positions)) targetOffsets.set(edgeId, offset);
  }

  return {
    nodeWidth,
    nodeHeight,
    width: margin * 2 + phases.length * nodeWidth + Math.max(phases.length - 1, 0) * columnGap,
    height: margin * 2 + maxColumnHeight,
    positions,
    sourceOffsets,
    targetOffsets,
  };
}

function edgePath(edge, layout, laneOffset = 0) {
  const from = layout.positions.get(edge.from);
  const to = layout.positions.get(edge.to);
  if (!from || !to) return '';
  const sourceX = from.x + layout.nodeWidth;
  const sourceY = from.y + layout.nodeHeight / 2 + (layout.sourceOffsets.get(edge.id) || 0) + laneOffset;
  const targetX = to.x;
  const targetY = to.y + layout.nodeHeight / 2 + (layout.targetOffsets.get(edge.id) || 0) + laneOffset;

  if (edge.from === edge.to) {
    const loop = 42;
    const top = Math.max(8, from.y - 20);
    return `M ${sourceX} ${sourceY} C ${sourceX + loop} ${sourceY}, ${sourceX + loop} ${top}, ${from.x + layout.nodeWidth / 2} ${top} C ${from.x - loop} ${top}, ${from.x - loop} ${targetY}, ${targetX} ${targetY}`;
  }

  if (targetX > sourceX) {
    const distance = Math.max(targetX - sourceX, 70);
    const curve = Math.min(Math.max(distance * 0.42, 48), 118);
    return `M ${sourceX} ${sourceY} C ${sourceX + curve} ${sourceY}, ${targetX - curve} ${targetY}, ${targetX} ${targetY}`;
  }

  const laneX = Math.max(sourceX, targetX) + 64 + Math.min(90, Math.abs(sourceY - targetY) * 0.22);
  return `M ${sourceX} ${sourceY} C ${laneX} ${sourceY}, ${laneX} ${targetY}, ${targetX} ${targetY}`;
}

function edgeLabel(edge, layout, view, options = {}) {
  const labelMode = options.labelMode || 'count';
  let label = '';
  let title = '';
  if (labelMode === 'action') {
    const action = edge.label || '(no action recorded)';
    label = `${compactInline(action, 34)}${edge.count > 1 ? ` ×${formatNumber(edge.count)}` : ''}`;
    title = `${action} · ${formatNumber(edge.count)} transitions`;
  } else {
    if (edge.count <= 1) return '';
    if (view.mode === 'all' && edge.count < Math.max(2, Math.ceil((view.totalTrajectories || 1) * 0.5))) return '';
    label = formatNumber(edge.count);
    title = `${formatNumber(edge.count)} transitions`;
  }
  const from = layout.positions.get(edge.from);
  const to = layout.positions.get(edge.to);
  if (!from || !to) return '';
  const sourceX = from.x + layout.nodeWidth;
  const sourceY = from.y + layout.nodeHeight / 2 + (layout.sourceOffsets.get(edge.id) || 0);
  const targetX = to.x;
  const targetY = to.y + layout.nodeHeight / 2 + (layout.targetOffsets.get(edge.id) || 0);
  const classes = ['graph-edge-label'];
  if (labelMode === 'action') classes.push('state-edge-label');
  return `<text class="${classes.join(' ')}" x="${(sourceX + targetX) / 2}" y="${(sourceY + targetY) / 2 - 5}"><title>${escapeHtml(title)}</title>${escapeHtml(label)}</text>`;
}

function renderNodeOrderBadge(orderLabels = []) {
  if (!orderLabels.length) return '';
  const label = orderLabels.join(',');
  return `<span class="graph-node-order" title="${escapeHtml(`Selected trajectory order: ${label}`)}">${escapeHtml(label)}</span>`;
}

function renderGraphNode(node, layout, focusedNodeId, selectedNodeId, highlightedNodeIds, hasHighlightedTrajectory, orderLabels = []) {
  const position = layout.positions.get(node.id);
  const classes = ['graph-node', node.kind];
  if (focusedNodeId === node.id) classes.push('focused');
  if (selectedNodeId === node.id) classes.push('selected');
  if (node.kind === 'state' && node.outgoing.length > 1) classes.push('branching');
  if (hasHighlightedTrajectory) classes.push(highlightedNodeIds.has(node.id) ? 'on-trajectory' : 'dimmed');
  const trajectoryList = (node.trajectoryIds || []).slice(0, 12).join(', ');
  const nodeText = node.description || node.label;
  const branchBadge = node.kind === 'state' && node.outgoing.length > 1 ? `<span class="graph-node-branch">branch ${formatNumber(node.outgoing.length)}</span>` : '';
  const title = `${nodeText}\n${formatNumber(node.occurrenceCount || node.attempts)} visits · ${formatNumber(node.attempts)} trajectories${trajectoryList ? `\nTrajectories: ${trajectoryList}` : ''}`;
  return `<button type="button" class="${classes.map(escapeHtml).join(' ')}" data-node-id="${escapeHtml(node.id)}" title="${escapeHtml(title)}" style="left:${position.x}px;top:${position.y}px;width:${layout.nodeWidth}px;height:${layout.nodeHeight}px">
    ${renderNodeOrderBadge(orderLabels)}
    <span class="graph-node-title">${escapeHtml(node.label)}</span>
    ${branchBadge}
  </button>`;
}

function renderNodeDetail(graph, view, selectedNodeId, highlightedPath, orderLabelMap) {
  const selectedNode = selectedNodeId ? graph.nodes.find((node) => node.id === selectedNodeId) : null;
  if (!selectedNode) {
    return `<div class="node-detail-panel empty">
      <strong>Node detail</strong>
      <span>点击一个 action 节点查看详情；点击一条 trajectory 后，节点右上角会显示真实 step 顺序。</span>
    </div>`;
  }

  const inCurrentView = view.nodes.some((node) => node.id === selectedNode.id);
  const statusText = statusEntries(selectedNode.statusCounts)
    .map(([status, count]) => `${statusBadge(status)} <span class="count">${formatNumber(count)}</span>`)
    .join(' ');
  const branchTags = [
    selectedNode.incoming.length > 1 ? `merge ${selectedNode.incoming.length}` : '',
    selectedNode.outgoing.length > 1 ? `branch ${selectedNode.outgoing.length}` : '',
  ].filter(Boolean);
  const trajectoryList = selectedNode.trajectoryIds.length
    ? selectedNode.trajectoryIds.slice(0, 16).map((id) => `<code>${escapeHtml(id)}</code>`).join(' ')
    : '<span class="muted">none</span>';
  const orderLabels = orderLabelMap.get(selectedNode.id) || [];
  const highlightedOrder = highlightedPath
    ? orderLabels.length
      ? `${trajectoryLabel(highlightedPath)} order: ${orderLabels.join(', ')}`
      : `${trajectoryLabel(highlightedPath)} does not visit this node in the current view.`
    : '选择一条 trajectory 后，节点角标和这里都会显示它访问该节点的顺序。';
  const terminalReasons = Object.entries(selectedNode.terminalReasons || {})
    .map(([reason, count]) => `<span>${escapeHtml(reason)} × ${formatNumber(count)}</span>`)
    .join(' ');

  return `<div class="node-detail-panel">
    <div class="node-detail-head">
      <div>
        <span class="muted">Selected node</span>
        <strong>${escapeHtml(selectedNode.label)}</strong>
      </div>
      <button type="button" data-clear-node>Clear node</button>
    </div>
    <div class="node-detail-grid">
      <div><span>Visits</span><strong>${formatNumber(selectedNode.occurrenceCount || selectedNode.attempts)}</strong></div>
      <div><span>Traj</span><strong>${formatNumber(selectedNode.attempts)}</strong></div>
      <div><span>First step</span><strong>${selectedNode.firstStep === null ? '—' : formatNumber(selectedNode.firstStep)}</strong></div>
      <div><span>Avg step</span><strong>${formatNumber(selectedNode.avgStep, 1)}</strong></div>
    </div>
    <div class="node-detail-row"><span>Status</span><div>${statusText || '<span class="muted">none</span>'}</div></div>
    <div class="node-detail-row"><span>Order</span><div>${escapeHtml(highlightedOrder)}</div></div>
    <div class="node-detail-row"><span>Shape</span><div>${branchTags.length ? branchTags.map((tag) => `<span class="badge">${escapeHtml(tag)}</span>`).join(' ') : '<span class="muted">linear</span>'}</div></div>
    ${terminalReasons ? `<div class="node-detail-row"><span>Terminal</span><div>${terminalReasons}</div></div>` : ''}
    <div class="node-detail-row"><span>Trajectories</span><div class="node-detail-trajectories">${trajectoryList}</div></div>
    <div class="node-detail-actions">
      <button type="button" data-focus-selected-node ${inCurrentView ? '' : 'disabled'}>Focus transitions</button>
      <span class="muted">图中节点保留 action 和顺序角标，边不显示 step 编号。</span>
    </div>
  </div>`;
}

function renderTrajectoryLegend(view) {
  const paths = view.paths || [];
  if (!paths.length) return '';
  const highlightedPath = graphHighlightedTrajectoryId ? paths.find((path) => path.id === graphHighlightedTrajectoryId) : null;
  const body = paths
    .map((path) => {
      const active = graphHighlightedTrajectoryId === path.id;
      const unavailable = path.visibleEdgeIds.length === 0;
      const classes = ['trajectory-chip', path.status || 'unknown'];
      if (active) classes.push('active');
      if (graphHighlightedTrajectoryId && !active) classes.push('dimmed');
      if (unavailable) classes.push('unavailable');
      const repeat = path.repeatId === null || path.repeatId === undefined ? '' : ` · repeat ${formatNumber(path.repeatId)}`;
      const scope = path.isComplete ? `${formatNumber(path.stepCount)} steps` : `${formatNumber(path.visibleEdgeIds.length)}/${formatNumber(path.edgeIds.length)} visible`;
      return `<button type="button" class="${classes.map(escapeHtml).join(' ')}" data-trajectory-id="${escapeHtml(path.id)}" style="--trajectory-color:${trajectoryColor(path.index)}" title="${escapeHtml(path.id)}">
        <span class="trajectory-swatch"></span>
        <strong>${escapeHtml(trajectoryLabel(path))}</strong>
        ${statusBadge(path.status)}
        <span>${escapeHtml(scope)}${escapeHtml(repeat)}</span>
        <span class="mono">${escapeHtml(compactTrajectoryId(path.id))}</span>
      </button>`;
    })
    .join('');
  const note = highlightedPath
    ? `Highlighted ${trajectoryLabel(highlightedPath)}; click another trajectory or clear to compare.`
    : 'Each colored line is one sampled trajectory. Click a trajectory to highlight its full path.';
  return `<div class="trajectory-legend">
    <div class="trajectory-legend-head">
      <strong>Trajectory paths</strong>
      <span>${escapeHtml(note)}</span>
      <button type="button" data-clear-trajectory ${highlightedPath ? '' : 'disabled'}>Clear trajectory</button>
    </div>
    <div class="trajectory-list">${body}</div>
  </div>`;
}

function renderSelectedSequence(path, nodeById) {
  if (!path) return '';
  const chips = path.nodeIds
    .map((nodeId, index) => {
      const node = nodeById.get(nodeId);
      const text = index === 0 ? 'START' : index === path.nodeIds.length - 1 ? path.status : (node?.label || nodeId.replace(/^action:/, ''));
      return `<span class="sequence-chip ${index === 0 || index === path.nodeIds.length - 1 ? 'terminal' : ''}"><span>${escapeHtml(text)}</span></span>`;
    })
    .join('<span class="sequence-arrow">→</span>');
  return `<div class="selected-sequence">
    <div class="selected-sequence-head"><strong>${escapeHtml(trajectoryLabel(path))} path</strong><span>重复 action 会回到同一个图节点；图中节点角标显示真实 step 顺序。</span></div>
    <div class="sequence-scroll">${chips}</div>
  </div>`;
}

function renderGraphToolbar(graph, view) {
  const focusedNode = view.focusedNodeId ? graph.nodes.find((node) => node.id === view.focusedNodeId) : null;
  const highlightedPath = graphHighlightedTrajectoryId ? view.paths.find((path) => path.id === graphHighlightedTrajectoryId) : null;
  const scope = focusedNode
    ? `Focused: ${focusedNode.label}`
    : highlightedPath
      ? `Highlighted ${trajectoryLabel(highlightedPath)} · ${highlightedPath.status} · ${formatNumber(highlightedPath.stepCount)} steps.`
    : graphViewMode === 'all'
      ? 'Showing every sampled trajectory as a colored path over shared action nodes.'
      : `Major transitions: edges seen in at least ${formatNumber(view.minMainEdgeCount)} attempts.`;
  return `<div class="graph-toolbar">
    <div class="graph-actions">
      <button type="button" data-graph-mode="all" class="${!focusedNode && graphViewMode === 'all' ? 'active' : ''}">Trajectory paths</button>
      <button type="button" data-graph-mode="overview" class="${!focusedNode && graphViewMode !== 'all' ? 'active' : ''}">Major transitions</button>
      <button type="button" data-clear-focus ${focusedNode ? '' : 'disabled'}>Clear focus</button>
    </div>
    <div class="graph-scope">${escapeHtml(scope)}</div>
  </div>`;
}

function bindGraphEvents(treeWrap) {
  treeWrap.querySelectorAll('[data-graph-mode]').forEach((button) => {
    button.addEventListener('click', () => {
      graphViewMode = button.dataset.graphMode === 'all' ? 'all' : 'overview';
      graphFocusNodeId = '';
      drawActionGraph(currentActionGraph);
    });
  });
  const clearFocus = treeWrap.querySelector('[data-clear-focus]');
  clearFocus?.addEventListener('click', () => {
    graphFocusNodeId = '';
    drawActionGraph(currentActionGraph);
  });
  const clearNode = treeWrap.querySelector('[data-clear-node]');
  clearNode?.addEventListener('click', () => {
    graphSelectedNodeId = '';
    drawActionGraph(currentActionGraph);
  });
  const focusSelectedNode = treeWrap.querySelector('[data-focus-selected-node]');
  focusSelectedNode?.addEventListener('click', () => {
    graphFocusNodeId = graphSelectedNodeId;
    drawActionGraph(currentActionGraph);
  });
  const clearTrajectory = treeWrap.querySelector('[data-clear-trajectory]');
  clearTrajectory?.addEventListener('click', () => {
    graphHighlightedTrajectoryId = '';
    drawActionGraph(currentActionGraph);
  });
  treeWrap.querySelectorAll('[data-trajectory-id]').forEach((button) => {
    button.addEventListener('click', () => {
      const trajectoryId = button.dataset.trajectoryId || '';
      graphHighlightedTrajectoryId = graphHighlightedTrajectoryId === trajectoryId ? '' : trajectoryId;
      drawActionGraph(currentActionGraph);
    });
  });
  treeWrap.querySelectorAll('[data-node-id]').forEach((button) => {
    button.addEventListener('click', () => {
      graphSelectedNodeId = button.dataset.nodeId || '';
      drawActionGraph(currentActionGraph);
    });
  });
}

function drawActionGraph(graph) {
  const treeWrap = $('trajectoryTree');
  if (!graph) return;
  const view = selectActionGraphView(graph, { mode: graphViewMode, focusedNodeId: graphFocusNodeId });
  graphFocusNodeId = view.focusedNodeId;
  view.totalTrajectories = graph.trajectories;
  let highlightedPath = graphHighlightedTrajectoryId ? view.paths.find((path) => path.id === graphHighlightedTrajectoryId) : null;
  if (graphHighlightedTrajectoryId && !highlightedPath) {
    graphHighlightedTrajectoryId = '';
    highlightedPath = null;
  }
  if (graphSelectedNodeId && !graph.nodes.some((node) => node.id === graphSelectedNodeId)) graphSelectedNodeId = '';
  const orderPhaseMap = highlightedPath ? nodeOrderPhaseMapForPath(highlightedPath, { visibleNodeIds: highlightedPath.visibleNodeIds }) : new Map();
  const layout = layoutGraph(view, orderPhaseMap);
  const maxEdgeCount = Math.max(...view.edges.map((edge) => edge.count), 1);
  const visibleActionNodes = view.nodes.filter((node) => node.kind === 'action').length;
  const totalActionNodes = graph.nodes.filter((node) => node.kind === 'action').length;
  const nodeById = new Map(view.nodes.map((node) => [node.id, node]));
  const graphNodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const highlightedNodeIds = new Set(highlightedPath?.visibleNodeIds || []);
  const orderLabelMap = new Map(
    highlightedPath
      ? nodeOrderEntriesForPath(highlightedPath, { visibleNodeIds: highlightedPath.visibleNodeIds }).map((entry) => [entry.nodeId, entry.labels])
      : [],
  );
  const baseEdges = view.edges
    .map((edge) => {
      const width = 1 + (edge.count / maxEdgeCount) * 2.4;
      const opacity = highlightedPath ? 0.04 : view.mode === 'all' ? 0.12 : 0.26 + (edge.count / maxEdgeCount) * 0.28;
      const classes = ['graph-edge', 'graph-edge-base', edge.to.startsWith('end:') ? 'terminal' : '', view.mode === 'focus' ? 'focus-edge' : ''].filter(Boolean).join(' ');
      return `<path class="${classes}" d="${edgePath(edge, layout)}" style="stroke-width:${width};opacity:${opacity}"><title>${escapeHtml(edge.from)} → ${escapeHtml(edge.to)} · ${formatNumber(edge.count)} transitions</title></path>`;
    })
    .join('');
  const trajectoryEdges = trajectorySegmentsForView(view, { laneStep: view.mode === 'all' ? 6 : 5 })
    .map((segment) => {
      const active = !highlightedPath || segment.trajectoryId === highlightedPath.id;
      if (highlightedPath && !active) return '';
      const classes = ['graph-edge', 'trajectory-edge', active ? 'active' : 'dimmed', segment.pathComplete ? '' : 'partial'].filter(Boolean).join(' ');
      const width = active ? (highlightedPath ? 3.3 : 2.35) : 1.1;
      const opacity = active ? (highlightedPath ? 0.96 : 0.78) : 0.12;
      const fromLabel = nodeById.get(segment.from)?.label || segment.from;
      const toLabel = nodeById.get(segment.to)?.label || segment.to;
      const title = `${trajectoryLabel({ index: segment.trajectoryIndex })} ${segment.trajectoryId} · ${fromLabel} → ${toLabel}`;
      return `<path class="${classes}" d="${edgePath(segment.edge, layout, segment.laneOffset)}" style="stroke:${trajectoryColor(segment.trajectoryIndex)};stroke-width:${width};opacity:${opacity}"><title>${escapeHtml(title)}</title></path>`;
    })
    .join('');
  const edgeLabels = view.mode === 'all' ? '' : view.edges.map((edge) => edgeLabel(edge, layout, view)).join('');

  const help = view.mode === 'overview'
    ? `高频转移视图会隐藏低频边；如果要逐条核对采样轨迹，请切回 Trajectory paths。当前显示 ${formatNumber(visibleActionNodes)}/${formatNumber(totalActionNodes)} 个 action 节点、${formatNumber(view.edges.length)}/${formatNumber(graph.edges.length)} 条转移。`
    : view.mode === 'focus'
      ? `聚焦模式只显示所选 action 的 incoming/outgoing 转移；轨迹图例会标出每条轨迹在当前焦点中可见的片段。当前隐藏 ${formatNumber(view.hiddenNodes)} 个节点、${formatNumber(view.hiddenEdges)} 条转移。`
      : `每条彩色线是一条采样轨迹，相同 action 仍合并成同一个节点；点击图例中的轨迹可高亮它的完整路径。当前显示全部 ${formatNumber(view.paths.length)} 条轨迹。`;

  treeWrap.className = 'graph-wrap';
  treeWrap.innerHTML = `${renderGraphToolbar(graph, view)}
    <div class="graph-help">${escapeHtml(help)}</div>
    ${renderTrajectoryLegend(view)}
    ${renderNodeDetail(graph, view, graphSelectedNodeId, highlightedPath, orderLabelMap)}
    ${renderSelectedSequence(highlightedPath, graphNodeById)}
    <div class="graph-canvas" style="width:${layout.width}px;height:${layout.height}px">
      <svg class="graph-edges" width="${layout.width}" height="${layout.height}" viewBox="0 0 ${layout.width} ${layout.height}" aria-hidden="true">
        <defs><marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs>
        ${baseEdges}${trajectoryEdges}${edgeLabels}
      </svg>
      ${view.nodes.map((node) => renderGraphNode(node, layout, view.focusedNodeId, graphSelectedNodeId, highlightedNodeIds, Boolean(highlightedPath), orderLabelMap.get(node.id) || [])).join('')}
    </div>`;
  bindGraphEvents(treeWrap);
}

function renderStateGraphToolbar(graph, view) {
  const focusedNode = view.focusedNodeId ? graph.nodes.find((node) => node.id === view.focusedNodeId) : null;
  const scope = focusedNode
    ? `Focused state: ${focusedNode.label}`
    : stateGraphViewMode === 'all'
      ? 'Showing observed states; edge labels are the actions taken from each state.'
      : `Major state transitions: edges seen in at least ${formatNumber(view.minMainEdgeCount)} attempts.`;
  return `<div class="graph-toolbar">
    <div class="graph-actions">
      <button type="button" data-state-graph-mode="all" class="${!focusedNode && stateGraphViewMode === 'all' ? 'active' : ''}">All states</button>
      <button type="button" data-state-graph-mode="overview" class="${!focusedNode && stateGraphViewMode !== 'all' ? 'active' : ''}">Major transitions</button>
      <button type="button" data-clear-state-focus ${focusedNode ? '' : 'disabled'}>Clear focus</button>
    </div>
    <div class="graph-scope">${escapeHtml(scope)}</div>
  </div>`;
}

function renderStateBranchList(graph) {
  const edgeById = new Map(graph.edges.map((edge) => [edge.id, edge]));
  const branchNodes = graph.nodes
    .filter((node) => node.kind === 'state' && node.outgoing.length > 1)
    .sort((a, b) => b.outgoing.length - a.outgoing.length || b.attempts - a.attempts || a.label.localeCompare(b.label))
    .slice(0, 12);

  if (!branchNodes.length) {
    return `<div class="state-branch-list empty">
      <strong>State branch points</strong>
      <span>No observation state produced multiple next actions in this sample.</span>
    </div>`;
  }

  const body = branchNodes
    .map((node, index) => {
      const actions = node.outgoing
        .map((edgeId) => edgeById.get(edgeId))
        .filter(Boolean)
        .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label))
        .map((edge) => `${edge.label || '(no action recorded)'}${edge.count > 1 ? ` ×${formatNumber(edge.count)}` : ''}`);
      return `<button type="button" class="state-branch-item" data-node-id="${escapeHtml(node.id)}" title="${escapeHtml(node.description || node.label)}">
        <strong>S${formatNumber(index + 1)}</strong>
        <span class="state-branch-text">${escapeHtml(node.label)}</span>
        <span class="state-branch-actions">${actions.slice(0, 5).map((action) => `<code>${escapeHtml(action)}</code>`).join(' ')}</span>
      </button>`;
    })
    .join('');

  return `<div class="state-branch-list">
    <div class="state-branch-head">
      <strong>State branch points</strong>
      <span>这些 observation 节点产生了多个不同 action；点击可聚焦。</span>
    </div>
    <div class="state-branch-items">${body}</div>
  </div>`;
}

function renderStateFocusDetail(graph, view) {
  const focusedNode = view.focusedNodeId ? graph.nodes.find((node) => node.id === view.focusedNodeId) : null;
  if (!focusedNode) return '';
  const edgeById = new Map(graph.edges.map((edge) => [edge.id, edge]));
  const outgoing = focusedNode.outgoing
    .map((edgeId) => edgeById.get(edgeId))
    .filter(Boolean)
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
  const incoming = focusedNode.incoming
    .map((edgeId) => edgeById.get(edgeId))
    .filter(Boolean)
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
  const statusText = statusEntries(focusedNode.statusCounts)
    .map(([status, count]) => `${statusBadge(status)} <span class="count">${formatNumber(count)}</span>`)
    .join(' ');

  return `<div class="state-detail-panel">
    <div class="node-detail-head">
      <div>
        <span class="muted">Focused state</span>
        <strong>${escapeHtml(focusedNode.label)}</strong>
      </div>
      <button type="button" data-clear-state-focus>Clear focus</button>
    </div>
    <div class="node-detail-grid">
      <div><span>Visits</span><strong>${formatNumber(focusedNode.occurrenceCount || focusedNode.attempts)}</strong></div>
      <div><span>Traj</span><strong>${formatNumber(focusedNode.attempts)}</strong></div>
      <div><span>Outgoing</span><strong>${formatNumber(focusedNode.outgoing.length)}</strong></div>
      <div><span>Incoming</span><strong>${formatNumber(focusedNode.incoming.length)}</strong></div>
    </div>
    <div class="node-detail-row"><span>Status</span><div>${statusText || '<span class="muted">none</span>'}</div></div>
    <div class="node-detail-row"><span>Next actions</span><div>${outgoing.length ? outgoing.map((edge) => `<code>${escapeHtml(edge.label || '(no action recorded)')}</code> <span class="muted">×${formatNumber(edge.count)}</span>`).join(' ') : '<span class="muted">terminal</span>'}</div></div>
    <div class="node-detail-row"><span>Prev actions</span><div>${incoming.length ? incoming.map((edge) => `<code>${escapeHtml(edge.label || '(no action recorded)')}</code> <span class="muted">×${formatNumber(edge.count)}</span>`).join(' ') : '<span class="muted">start</span>'}</div></div>
    <details class="state-observation-detail" open>
      <summary>Full observation</summary>
      <pre>${escapeHtml(focusedNode.description || focusedNode.label)}</pre>
    </details>
  </div>`;
}

function bindStateGraphEvents(treeWrap) {
  treeWrap.querySelectorAll('[data-state-graph-mode]').forEach((button) => {
    button.addEventListener('click', () => {
      stateGraphViewMode = button.dataset.stateGraphMode === 'all' ? 'all' : 'overview';
      stateGraphFocusNodeId = '';
      drawStateGraph(currentStateGraph);
    });
  });
  treeWrap.querySelectorAll('[data-clear-state-focus]').forEach((button) => {
    button.addEventListener('click', () => {
      stateGraphFocusNodeId = '';
      drawStateGraph(currentStateGraph);
    });
  });
  treeWrap.querySelectorAll('[data-node-id]').forEach((button) => {
    button.addEventListener('click', () => {
      stateGraphFocusNodeId = button.dataset.nodeId || '';
      drawStateGraph(currentStateGraph);
    });
  });
}

function drawStateGraph(graph) {
  const treeWrap = $('stateTransitionTree');
  if (!graph || !treeWrap) return;
  const view = selectActionGraphView(graph, { mode: stateGraphViewMode, focusedNodeId: stateGraphFocusNodeId });
  stateGraphFocusNodeId = view.focusedNodeId;
  view.totalTrajectories = graph.trajectories;
  const layout = layoutGraph(view, new Map(), { nodeWidth: 220, nodeHeight: 64, columnGap: 132, rowGap: 26 });
  const maxEdgeCount = Math.max(...view.edges.map((edge) => edge.count), 1);
  const visibleStateNodes = view.nodes.filter((node) => node.kind === 'state').length;
  const totalStateNodes = graph.nodes.filter((node) => node.kind === 'state').length;
  const baseEdges = view.edges
    .map((edge) => {
      const width = 1.4 + (edge.count / maxEdgeCount) * 3.2;
      const opacity = view.mode === 'all' ? 0.62 : 0.72;
      const classes = ['graph-edge', 'state-transition-edge', edge.to.startsWith('end:') ? 'terminal' : '', view.mode === 'focus' ? 'focus-edge' : ''].filter(Boolean).join(' ');
      const title = `${edge.label || '(no action recorded)'} · ${edge.from} → ${edge.to} · ${formatNumber(edge.count)} transitions`;
      return `<path class="${classes}" d="${edgePath(edge, layout)}" style="stroke-width:${width};opacity:${opacity}"><title>${escapeHtml(title)}</title></path>`;
    })
    .join('');
  const edgeLabels = view.edges.map((edge) => edgeLabel(edge, layout, view, { labelMode: 'action' })).join('');
  const help = view.mode === 'overview'
    ? `高频状态转移视图会隐藏低频 action 边；当前显示 ${formatNumber(visibleStateNodes)}/${formatNumber(totalStateNodes)} 个 state 节点、${formatNumber(view.edges.length)}/${formatNumber(graph.edges.length)} 条 action 边。`
    : view.mode === 'focus'
      ? `聚焦模式只显示所选 observation state 的 incoming/outgoing action 边。当前隐藏 ${formatNumber(view.hiddenNodes)} 个节点、${formatNumber(view.hiddenEdges)} 条边。`
      : `节点是 observation/state，边标签是从该 state 采取的 action；黄色 branch 节点表示同一 state 后面出现了多个不同 action。`;

  treeWrap.className = 'graph-wrap state-graph';
  treeWrap.innerHTML = `${renderStateGraphToolbar(graph, view)}
    <div class="graph-help">${escapeHtml(help)}</div>
    ${renderStateBranchList(graph)}
    ${renderStateFocusDetail(graph, view)}
    <div class="graph-canvas" style="width:${layout.width}px;height:${layout.height}px">
      <svg class="graph-edges" width="${layout.width}" height="${layout.height}" viewBox="0 0 ${layout.width} ${layout.height}" aria-hidden="true">
        <defs><marker id="graph-arrow-state" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs>
        ${baseEdges}${edgeLabels}
      </svg>
      ${view.nodes.map((node) => renderGraphNode(node, layout, view.focusedNodeId, '', new Set(), false)).join('')}
    </div>`;
  bindStateGraphEvents(treeWrap);
}

function renderActionGraph(rows) {
  const treeSummary = $('treeSummary');
  const treeWrap = $('trajectoryTree');
  graphViewMode = 'all';
  graphFocusNodeId = '';
  graphHighlightedTrajectoryId = '';
  graphSelectedNodeId = '';
  currentActionGraph = null;
  if (!rows.length) {
    treeSummary.className = 'tree-summary empty';
    treeSummary.textContent = 'No trajectory rows were found for this sample.';
    treeWrap.className = 'graph-wrap empty';
    treeWrap.textContent = 'No actions recorded.';
    return;
  }

  const graph = buildActionGraph(rows);
  currentActionGraph = graph;
  const summary = summarizeActionGraph(graph);
  treeSummary.className = 'tree-summary';
  treeSummary.innerHTML = [
    ['Attempts', formatNumber(summary.trajectories)],
    ['Unique actions', formatNumber(summary.uniqueActions)],
    ['Transitions', formatNumber(summary.transitionEdges)],
    ['Branches', formatNumber(summary.branchPoints)],
    ['Convergences', formatNumber(summary.convergencePoints)],
    ['Max length', formatNumber(summary.maxPathLength)],
  ]
    .map(([label, value]) => `<div class="tree-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join('');
  drawActionGraph(graph);
}

function renderStateGraph(rows) {
  const stateTreeSummary = $('stateTreeSummary');
  const stateTreeWrap = $('stateTransitionTree');
  stateGraphViewMode = 'all';
  stateGraphFocusNodeId = '';
  currentStateGraph = null;
  if (!stateTreeSummary || !stateTreeWrap) return;
  if (!rows.length) {
    stateTreeSummary.className = 'tree-summary empty';
    stateTreeSummary.textContent = 'No trajectory rows were found for this sample.';
    stateTreeWrap.className = 'graph-wrap empty';
    stateTreeWrap.textContent = 'No observations recorded.';
    return;
  }

  const graph = buildStateGraph(rows);
  currentStateGraph = graph;
  const summary = summarizeStateGraph(graph);
  stateTreeSummary.className = 'tree-summary';
  stateTreeSummary.innerHTML = [
    ['Attempts', formatNumber(summary.trajectories)],
    ['Unique states', formatNumber(summary.uniqueStates)],
    ['Action edges', formatNumber(summary.actionEdges)],
    ['State branches', formatNumber(summary.branchPoints)],
    ['Convergences', formatNumber(summary.convergencePoints)],
    ['Max length', formatNumber(summary.maxPathLength)],
  ]
    .map(([label, value]) => `<div class="tree-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join('');
  drawStateGraph(graph);
}

async function loadPage() {
  if (!sampleId) {
    showMessage('Missing sample id in URL.', 'error');
    return;
  }

  $('sampleCards').innerHTML = '';
  $('attemptsTable').innerHTML = '<tbody><tr><td class="empty">Loading sample attempts...</td></tr></tbody>';
  $('treeSummary').textContent = 'Loading action graph...';
  $('trajectoryTree').textContent = 'Reading full trajectory rows...';
  if ($('stateTreeSummary')) $('stateTreeSummary').textContent = 'Loading state graph...';
  if ($('stateTransitionTree')) $('stateTransitionTree').textContent = 'Reading full trajectory rows...';

  try {
    const datasetPayload = await ensureDatasetLoaded((text) => showMessage(text, 'info'));
    setServerStatus(`${datasetPayload.summary.totalTrajectories.toLocaleString()} trajectories loaded`);
    const payload = await api(`/api/sample-detail?id=${encodeURIComponent(sampleId)}`);
    renderSample(payload.sample, payload.trajectories);
    renderActionGraph(payload.rows || []);
    renderStateGraph(payload.rows || []);
    showMessage(`Loaded sample ${sampleId}: ${payload.trajectories.length} trajectory attempts.`, 'info');
  } catch (error) {
    setServerStatus('No dataset loaded');
    const hint = error.status === 409 ? ' Go back to Overview and load a directory first.' : '';
    showMessage(`${error.message}${hint}`, 'error');
    $('sampleCards').innerHTML = '';
    $('treeSummary').className = 'tree-summary empty';
    $('treeSummary').textContent = 'Action graph unavailable.';
    $('trajectoryTree').className = 'graph-wrap empty';
    $('trajectoryTree').textContent = error.message;
    if ($('stateTreeSummary')) {
      $('stateTreeSummary').className = 'tree-summary empty';
      $('stateTreeSummary').textContent = 'State graph unavailable.';
    }
    if ($('stateTransitionTree')) {
      $('stateTransitionTree').className = 'graph-wrap empty';
      $('stateTransitionTree').textContent = error.message;
    }
    $('attemptsTable').innerHTML = `<tbody><tr><td class="empty">${escapeHtml(error.message)}</td></tr></tbody>`;
  }
}

loadPage();
