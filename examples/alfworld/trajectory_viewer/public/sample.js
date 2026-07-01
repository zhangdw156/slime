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
import { buildActionGraph, selectActionGraphView, statusEntries, summarizeActionGraph } from './tree.js';

const sampleId = currentQuery().get('id') || '';
let currentActionGraph = null;
let graphViewMode = 'overview';
let graphFocusNodeId = '';

function compactPath(value) {
  const text = String(value || '');
  const parts = text.split('/').filter(Boolean);
  if (parts.length <= 4) return text;
  return `…/${parts.slice(-4).join('/')}`;
}

function renderSample(sample, trajectories) {
  $('sampleSubtitle').textContent = `Sample ${sample.sampleId} · ${sample.taskType} · ${sample.split}`;
  $('sampleCards').innerHTML = [
    card('Sample', sample.sampleId, compactPath(sample.gamefile)),
    card('Task', sample.taskType, sample.split),
    card('Attempts', formatNumber(sample.attempts), `${formatNumber(sample.successes)} successes`),
    card('Status', sample.success ? 'success@any' : 'no success', Object.entries(sample.statusCounts).map(([status, count]) => `${status}:${count}`).join(' ')),
    card('Best success', sample.bestSuccessId || 'none', sample.bestSuccessSteps !== null ? `${formatNumber(sample.bestSuccessSteps)} steps` : ''),
    card('Invalid actions', formatNumber(sample.invalidActionCount), `avg steps ${formatNumber(sample.avgSteps, 2)}`),
  ].join('');
  $('attemptsTable').innerHTML = renderTrajectoryTable(trajectories, { includeSample: false });
  const gamefilePath = $('gamefilePath');
  if (sample.gamefile) {
    gamefilePath.classList.remove('hidden');
    gamefilePath.innerHTML = `<strong>Gamefile</strong><code>${escapeHtml(sample.gamefile)}</code>`;
  } else {
    gamefilePath.classList.add('hidden');
    gamefilePath.textContent = '';
  }
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
  return group.map((edge, index) => [edge.id, (index - (group.length - 1) / 2) * 8]);
}

function layoutGraph(view) {
  const nodeWidth = 176;
  const nodeHeight = 70;
  const columnGap = 86;
  const rowGap = 18;
  const margin = 28;
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
    if (!grouped.has(node.phase)) grouped.set(node.phase, []);
    grouped.get(node.phase).push(node);
  }

  const phases = [...grouped.keys()].sort((a, b) => a - b);
  const order = new Map();
  const kindOrder = { start: 0, action: 1, end: 2 };
  const sortWithinPhase = (nodes) => nodes.sort((a, b) => kindOrder[a.kind] - kindOrder[b.kind] || b.attempts - a.attempts || b.occurrenceCount - a.occurrenceCount || a.label.localeCompare(b.label));
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

function edgePath(edge, layout) {
  const from = layout.positions.get(edge.from);
  const to = layout.positions.get(edge.to);
  if (!from || !to) return '';
  const sourceX = from.x + layout.nodeWidth;
  const sourceY = from.y + layout.nodeHeight / 2 + (layout.sourceOffsets.get(edge.id) || 0);
  const targetX = to.x;
  const targetY = to.y + layout.nodeHeight / 2 + (layout.targetOffsets.get(edge.id) || 0);

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

function edgeLabel(edge, layout, view) {
  if (edge.count <= 1) return '';
  const from = layout.positions.get(edge.from);
  const to = layout.positions.get(edge.to);
  if (!from || !to) return '';
  if (view.mode === 'all' && edge.count < Math.max(2, Math.ceil((view.totalTrajectories || 1) * 0.5))) return '';
  const sourceX = from.x + layout.nodeWidth;
  const sourceY = from.y + layout.nodeHeight / 2 + (layout.sourceOffsets.get(edge.id) || 0);
  const targetX = to.x;
  const targetY = to.y + layout.nodeHeight / 2 + (layout.targetOffsets.get(edge.id) || 0);
  return `<text class="graph-edge-label" x="${(sourceX + targetX) / 2}" y="${(sourceY + targetY) / 2 - 5}">${formatNumber(edge.count)}</text>`;
}

function renderGraphNode(node, layout, focusedNodeId) {
  const position = layout.positions.get(node.id);
  const meta = node.kind === 'action'
    ? `${formatNumber(node.occurrenceCount)} visits · ${formatNumber(node.attempts)} traj`
    : `${formatNumber(node.attempts)} traj`;
  const tags = [];
  if (node.kind === 'action' && node.outgoing.length > 1) tags.push(`branch ${node.outgoing.length}`);
  if (node.kind === 'action' && node.incoming.length > 1) tags.push(`merge ${node.incoming.length}`);
  const focusedClass = focusedNodeId === node.id ? ' focused' : '';
  return `<button type="button" class="graph-node ${escapeHtml(node.kind)}${focusedClass}" data-node-id="${escapeHtml(node.id)}" title="${escapeHtml(node.label)}" style="left:${position.x}px;top:${position.y}px;width:${layout.nodeWidth}px;height:${layout.nodeHeight}px">
    <span class="graph-node-title">${escapeHtml(node.label)}</span>
    <span class="graph-node-meta">${escapeHtml(meta)}</span>
    <span class="graph-node-footer">${tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join('')}${renderStatusCounts(node.statusCounts)}</span>
  </button>`;
}

function renderGraphToolbar(graph, view) {
  const focusedNode = view.focusedNodeId ? graph.nodes.find((node) => node.id === view.focusedNodeId) : null;
  const scope = focusedNode
    ? `Focused: ${focusedNode.label}`
    : graphViewMode === 'all'
      ? 'Showing full graph; this can be dense.'
      : `Readable overview: transitions seen in at least ${formatNumber(view.minMainEdgeCount)} attempts.`;
  return `<div class="graph-toolbar">
    <div class="graph-actions">
      <button type="button" data-graph-mode="overview" class="${!focusedNode && graphViewMode !== 'all' ? 'active' : ''}">Readable overview</button>
      <button type="button" data-graph-mode="all" class="${!focusedNode && graphViewMode === 'all' ? 'active' : ''}">All transitions</button>
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
  treeWrap.querySelectorAll('[data-node-id]').forEach((button) => {
    button.addEventListener('click', () => {
      graphFocusNodeId = button.dataset.nodeId || '';
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
  const layout = layoutGraph(view);
  const maxEdgeCount = Math.max(...view.edges.map((edge) => edge.count), 1);
  const visibleActionNodes = view.nodes.filter((node) => node.kind === 'action').length;
  const totalActionNodes = graph.nodes.filter((node) => node.kind === 'action').length;
  const paths = view.edges
    .map((edge) => {
      const width = view.mode === 'all' ? 0.8 + (edge.count / maxEdgeCount) * 2.8 : 1.3 + (edge.count / maxEdgeCount) * 4.2;
      const opacity = view.mode === 'all' ? 0.16 + (edge.count / maxEdgeCount) * 0.34 : 0.28 + (edge.count / maxEdgeCount) * 0.5;
      const classes = ['graph-edge', edge.to.startsWith('end:') ? 'terminal' : '', view.mode === 'focus' ? 'focus-edge' : ''].filter(Boolean).join(' ');
      return `<path class="${classes}" d="${edgePath(edge, layout)}" style="stroke-width:${width};opacity:${opacity}"><title>${escapeHtml(edge.from)} → ${escapeHtml(edge.to)} · ${formatNumber(edge.count)} transitions</title></path>${edgeLabel(edge, layout, view)}`;
    })
    .join('');

  const help = view.mode === 'overview'
    ? `默认隐藏低频转移以保证可读性；点击任意 action 可只看它的全部相邻转移。当前显示 ${formatNumber(visibleActionNodes)}/${formatNumber(totalActionNodes)} 个 action 节点、${formatNumber(view.edges.length)}/${formatNumber(graph.edges.length)} 条转移。`
    : view.mode === 'focus'
      ? `聚焦模式只显示所选 action 的 incoming/outgoing 转移。当前隐藏 ${formatNumber(view.hiddenNodes)} 个节点、${formatNumber(view.hiddenEdges)} 条转移。`
      : `完整图会保留所有节点和转移，样本较复杂时会比默认总览更密。`;

  treeWrap.className = 'graph-wrap';
  treeWrap.innerHTML = `${renderGraphToolbar(graph, view)}
    <div class="graph-help">${escapeHtml(help)}</div>
    <div class="graph-canvas" style="width:${layout.width}px;height:${layout.height}px">
      <svg class="graph-edges" width="${layout.width}" height="${layout.height}" viewBox="0 0 ${layout.width} ${layout.height}" aria-hidden="true">
        <defs><marker id="graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z"></path></marker></defs>
        ${paths}
      </svg>
      ${view.nodes.map((node) => renderGraphNode(node, layout, view.focusedNodeId)).join('')}
    </div>`;
  bindGraphEvents(treeWrap);
}

function renderActionGraph(rows) {
  const treeSummary = $('treeSummary');
  const treeWrap = $('trajectoryTree');
  graphViewMode = 'overview';
  graphFocusNodeId = '';
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

async function loadPage() {
  if (!sampleId) {
    showMessage('Missing sample id in URL.', 'error');
    return;
  }

  $('sampleCards').innerHTML = '';
  $('attemptsTable').innerHTML = '<tbody><tr><td class="empty">Loading sample attempts...</td></tr></tbody>';
  $('treeSummary').textContent = 'Loading action graph...';
  $('trajectoryTree').textContent = 'Reading full trajectory rows...';

  try {
    const datasetPayload = await ensureDatasetLoaded((text) => showMessage(text, 'info'));
    setServerStatus(`${datasetPayload.summary.totalTrajectories.toLocaleString()} trajectories loaded`);
    const payload = await api(`/api/sample-detail?id=${encodeURIComponent(sampleId)}`);
    renderSample(payload.sample, payload.trajectories);
    renderActionGraph(payload.rows || []);
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
    $('attemptsTable').innerHTML = `<tbody><tr><td class="empty">${escapeHtml(error.message)}</td></tr></tbody>`;
  }
}

loadPage();
