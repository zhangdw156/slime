import {
  $,
  anchor,
  api,
  card,
  escapeHtml,
  formatNumber,
  formatPercent,
  loadDataset,
  makePager,
  rememberedDirectory,
  renderTrajectoryTable,
  sampleHref,
  setServerStatus,
  showMessage,
  statusBadge,
  trajectoryHref,
} from './common.js';

const state = {
  summary: null,
  options: { statuses: [], taskTypes: [], splits: [] },
  directory: rememberedDirectory(),
  pages: { samples: 1, trajectories: 1, sft: 1 },
};

const directoryInput = $('directoryInput');
directoryInput.value = state.directory;

function setLoading(loading) {
  $('loadButton').disabled = loading;
  $('refreshButton').disabled = loading;
  $('browseButton').disabled = loading;
}

function fillSelect(select, values, label = 'All') {
  const current = select.value;
  select.innerHTML = `<option value="all">${escapeHtml(label)}</option>${values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join('')}`;
  if ([...select.options].some((option) => option.value === current)) select.value = current;
}

function syncFilterOptions() {
  const statuses = state.options.statuses || [];
  const taskTypes = state.options.taskTypes || [];
  const splits = state.options.splits || [];
  fillSelect($('trajectoryStatus'), statuses, 'All statuses');
  fillSelect($('trajectoryTaskType'), taskTypes, 'All task types');
  fillSelect($('trajectorySplit'), splits, 'All splits');
  fillSelect($('sampleStatus'), ['success', ...statuses.filter((item) => item !== 'success')], 'All sample states');
  fillSelect($('sampleTaskType'), taskTypes, 'All task types');
  fillSelect($('sampleSplit'), splits, 'All splits');
}

function renderOverview() {
  const summary = state.summary;
  if (!summary) {
    $('overviewCards').innerHTML = '';
    $('statusBars').textContent = 'Load a directory to see status counts.';
    $('metricList').textContent = 'No dataset loaded.';
    $('taskTypeTable').innerHTML = '';
    setServerStatus('No directory loaded');
    return;
  }

  setServerStatus(`${summary.totalTrajectories.toLocaleString()} trajectories loaded`);
  $('overviewCards').innerHTML = [
    card('Trajectories', formatNumber(summary.totalTrajectories), summary.fileName),
    card('Samples', formatNumber(summary.uniqueSamples), `${summary.completeSamples} complete @8`),
    card('Success rate', formatPercent(summary.successRate), 'per trajectory'),
    card('Success@any', formatPercent(summary.successAtAny), 'per sample'),
    card('Success@8', formatPercent(summary.successAt8Complete), 'complete samples only'),
    card('SFT candidates', formatNumber(summary.selectedSftSamples), 'shortest successful per sample'),
  ].join('');

  renderStatusBars(summary.statusCounts || {});
  renderMetrics(summary);
  renderTaskTypes(summary.taskTypeStats || []);
}

function renderStatusBars(statusCounts) {
  const entries = Object.entries(statusCounts).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    $('statusBars').textContent = 'No status counts.';
    return;
  }
  const max = Math.max(...entries.map(([, count]) => count), 1);
  $('statusBars').innerHTML = entries
    .map(([status, count]) => {
      const width = Math.max(2, (count / max) * 100);
      return `<div class="bar-row"><span class="badge ${escapeHtml(status)}">${escapeHtml(status)}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><span>${formatNumber(count)}</span></div>`;
    })
    .join('');
}

function renderMetrics(summary) {
  const metrics = [
    ['Avg steps', formatNumber(summary.avgSteps, 2)],
    ['Avg invalid actions', formatNumber(summary.avgInvalidActions, 2)],
    ['Avg prompt tokens', formatNumber(summary.avgPromptTokens, 1)],
    ['Avg response tokens', formatNumber(summary.avgResponseTokens, 1)],
    ['Parse errors', formatNumber(summary.parseErrorCount), summary.parseErrorCount ? 'check JSONL tail' : 'clean'],
    ['Indexed in', `${formatNumber(summary.indexDurationMs)} ms`],
  ];
  $('metricList').innerHTML = metrics
    .map(([label, value, sub]) => `<div class="metric"><span class="muted">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong>${sub ? `<span class="muted">${escapeHtml(sub)}</span>` : ''}</div>`)
    .join('');
}

function renderTaskTypes(taskTypes) {
  if (!taskTypes.length) {
    $('taskTypeTable').innerHTML = '<tbody><tr><td class="empty">No task type data.</td></tr></tbody>';
    return;
  }
  $('taskTypeTable').innerHTML = `
    <thead><tr><th>Task type</th><th>Trajectories</th><th>Samples</th><th>Successes</th><th>Success rate</th></tr></thead>
    <tbody>${taskTypes
      .map(
        (item) => `<tr><td>${escapeHtml(item.taskType)}</td><td>${formatNumber(item.trajectories)}</td><td>${formatNumber(item.samples)}</td><td>${formatNumber(item.successes)}</td><td>${formatPercent(item.successRate)}</td></tr>`,
      )
      .join('')}</tbody>`;
}

async function loadSamples() {
  if (!state.summary) return;
  const params = new URLSearchParams({
    page: state.pages.samples,
    pageSize: 50,
    q: $('sampleSearch').value,
    status: $('sampleStatus').value,
    taskType: $('sampleTaskType').value,
    split: $('sampleSplit').value,
  });
  const payload = await api(`/api/samples?${params}`);
  const rows = payload.items.length
    ? payload.items
        .map(
          (item) => `<tr><td class="mono">${anchor(sampleHref(item.sampleId), item.sampleId)}</td><td>${escapeHtml(item.taskType)}</td><td>${escapeHtml(item.split)}</td><td>${formatNumber(item.attempts)}</td><td>${Object.entries(item.statusCounts)
            .map(([status, count]) => `${statusBadge(status)} ${formatNumber(count)}`)
            .join(' ')}</td><td>${item.bestSuccessId ? `${anchor(trajectoryHref(item.bestSuccessId, item.sampleId), item.bestSuccessId, 'mono')}<div class="muted">${formatNumber(item.bestSuccessSteps)} steps</div>` : '<span class="muted">none</span>'}</td><td>${formatNumber(item.invalidActionCount)}</td></tr>`,
        )
        .join('')
    : '<tr><td colspan="7" class="empty">No samples match the current filters.</td></tr>';
  $('samplesTable').innerHTML = `
    <thead><tr><th>Sample</th><th>Task</th><th>Split</th><th>Attempts</th><th>Statuses</th><th>Best success</th><th>Invalid actions</th></tr></thead>
    <tbody>${rows}</tbody>`;
  $('samplesPager').innerHTML = makePager('samples', payload);
}

async function loadTrajectories() {
  if (!state.summary) return;
  const params = new URLSearchParams({
    page: state.pages.trajectories,
    pageSize: 50,
    q: $('trajectorySearch').value,
    status: $('trajectoryStatus').value,
    taskType: $('trajectoryTaskType').value,
    split: $('trajectorySplit').value,
  });
  const payload = await api(`/api/trajectories?${params}`);
  $('trajectoriesTable').innerHTML = renderTrajectoryTable(payload.items);
  $('trajectoriesPager').innerHTML = makePager('trajectories', payload);
}

async function loadSftCandidates() {
  if (!state.summary) return;
  const params = new URLSearchParams({ page: state.pages.sft, pageSize: 50 });
  const payload = await api(`/api/sft-candidates?${params}`);
  const rows = payload.items.length
    ? payload.items
        .map(
          (item) => `<tr><td>${anchor(trajectoryHref(item.id, item.sampleId), item.id, 'mono')}</td><td class="mono">${anchor(sampleHref(item.sampleId), item.sampleId)}</td><td>${escapeHtml(item.taskType)}</td><td>${formatNumber(item.numSteps)}</td><td>${formatNumber(item.responseTokens)}</td><td>${formatNumber(item.invalidActionCount)}</td></tr>`,
        )
        .join('')
    : '<tr><td colspan="6" class="empty">No successful SFT candidates are available.</td></tr>';
  $('sftTable').innerHTML = `
    <thead><tr><th>Selected trajectory</th><th>Sample</th><th>Task</th><th>Steps</th><th>Response tokens</th><th>Invalid actions</th></tr></thead>
    <tbody>${rows}</tbody>`;
  $('sftPager').innerHTML = makePager('sft', payload);
}

async function loadAllTables() {
  await Promise.all([loadSamples(), loadTrajectories(), loadSftCandidates()]);
}

function applyDatasetPayload(payload) {
  state.summary = payload.summary;
  state.options = payload.options;
  state.directory = payload.summary.directory;
  directoryInput.value = state.directory;
  syncFilterOptions();
  renderOverview();
}

async function loadDirectory() {
  const directory = directoryInput.value.trim();
  if (!directory) {
    showMessage('Enter a server directory containing all_trajectories.jsonl.', 'error');
    return;
  }
  setLoading(true);
  showMessage('Indexing all_trajectories.jsonl ... this may take a moment for large files.', 'info');
  try {
    const payload = await loadDataset(directory);
    state.pages = { samples: 1, trajectories: 1, sft: 1 };
    applyDatasetPayload(payload);
    await loadAllTables();
    showMessage(`Loaded ${payload.summary.totalTrajectories.toLocaleString()} trajectories from ${payload.summary.filePath}.`, 'info');
  } catch (error) {
    showMessage(error.message, 'error');
  } finally {
    setLoading(false);
  }
}

async function browse(directory = directoryInput.value.trim()) {
  try {
    const payload = await api(`/api/browse?dir=${encodeURIComponent(directory || '.')}`);
    directoryInput.value = payload.directory;
    const panel = $('browsePanel');
    panel.classList.remove('hidden');
    panel.innerHTML = `<div><strong>${escapeHtml(payload.directory)}</strong> ${payload.hasTrajectoryFile ? '<span class="badge success">has all_trajectories.jsonl</span>' : '<span class="badge failed">no all_trajectories.jsonl</span>'}</div><div class="browse-actions"><button data-browse-dir="${escapeHtml(payload.parent)}">..</button>${payload.children
      .map((child) => `<button data-browse-dir="${escapeHtml(`${payload.directory}/${child}`)}">${escapeHtml(child)}</button>`)
      .join('')}</div>`;
  } catch (error) {
    showMessage(error.message, 'error');
  }
}

function resetAndReload(kind) {
  state.pages[kind] = 1;
  if (kind === 'samples') loadSamples();
  if (kind === 'trajectories') loadTrajectories();
}

$('loadButton').addEventListener('click', loadDirectory);
$('refreshButton').addEventListener('click', loadDirectory);
$('browseButton').addEventListener('click', () => browse());
for (const id of ['sampleSearch', 'sampleStatus', 'sampleTaskType', 'sampleSplit']) {
  $(id).addEventListener('input', () => resetAndReload('samples'));
}
for (const id of ['trajectorySearch', 'trajectoryStatus', 'trajectoryTaskType', 'trajectorySplit']) {
  $(id).addEventListener('input', () => resetAndReload('trajectories'));
}

document.addEventListener('click', async (event) => {
  const target = event.target.closest('button');
  if (!target) return;
  if (target.dataset.pageKind) {
    state.pages[target.dataset.pageKind] = Number(target.dataset.page);
    if (target.dataset.pageKind === 'samples') await loadSamples();
    if (target.dataset.pageKind === 'trajectories') await loadTrajectories();
    if (target.dataset.pageKind === 'sft') await loadSftCandidates();
    return;
  }
  if (target.dataset.browseDir) await browse(target.dataset.browseDir);
});

async function restoreLoadedDataset() {
  try {
    const payload = await api('/api/summary');
    applyDatasetPayload(payload);
    await loadAllTables();
    showMessage(`Using loaded directory ${payload.summary.directory}.`, 'info');
  } catch (error) {
    if (error.status === 409 && state.directory) {
      showMessage('Previous directory restored. Click Load to index it.', 'info');
      return;
    }
    if (error.status !== 409) showMessage(error.message, 'error');
  }
}

restoreLoadedDataset();
