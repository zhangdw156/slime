import {
  $,
  anchor,
  api,
  card,
  currentQuery,
  ensureDatasetLoaded,
  escapeHtml,
  formatNumber,
  renderStepCard,
  sampleHref,
  setServerStatus,
  showMessage,
  statusBadge,
} from './common.js';

const query = currentQuery();
const trajectoryId = query.get('id') || '';
const requestedSampleId = query.get('sampleId') || '';

function compactPath(value) {
  const text = String(value || '');
  const parts = text.split('/').filter(Boolean);
  if (parts.length <= 4) return text;
  return `…/${parts.slice(-4).join('/')}`;
}

function renderTrajectory(row) {
  const id = row.trajectory_id || row.episode_id || trajectoryId;
  const sampleId = row.sample_id ?? requestedSampleId;
  const steps = Array.isArray(row.steps) ? row.steps : [];

  $('detailSubtitle').textContent = id;
  $('detailNav').innerHTML = `${anchor('/', '← Overview')} ${sampleId !== '' && sampleId !== undefined ? anchor(sampleHref(sampleId), `← Sample ${sampleId}`) : ''}`;
  $('trajectoryCards').innerHTML = [
    card('Trajectory', id, compactPath(row.gamefile)),
    card('Sample', sampleId, row.sample_repeat_id !== undefined ? `repeat ${row.sample_repeat_id}` : ''),
    card('Status', row.status || (row.won ? 'success' : 'unknown'), `reward ${row.reward ?? 0}`),
    card('Steps', formatNumber(row.num_steps ?? steps.length), row.terminal_reason || row.error || ''),
    card('Task', row.task_type || 'unknown', row.split || ''),
    card('Invalid actions', formatNumber(row.invalid_action_count), row.teacher_model || ''),
  ].join('');

  $('taskPanel').innerHTML = `
    <div class="section-heading"><div><h2>Task context</h2><p>High-level trajectory metadata.</p></div></div>
    <div class="detail-grid">
      <div class="metric"><span class="muted">Status</span><strong>${statusBadge(row.status || (row.won ? 'success' : 'unknown'))}</strong></div>
      <div class="metric"><span class="muted">Done / won</span><strong>${escapeHtml(row.done)} / ${escapeHtml(row.won)}</strong></div>
      <div class="metric"><span class="muted">Terminal reason</span><strong>${escapeHtml(row.terminal_reason || row.error || '')}</strong></div>
      <div class="metric"><span class="muted">Elapsed sec</span><strong>${formatNumber(row.elapsed_sec, 2)}</strong></div>
    </div>
    ${row.gamefile ? `<div class="path-note inline"><strong>Gamefile</strong><code>${escapeHtml(row.gamefile)}</code></div>` : ''}
    ${row.task_description ? `<h3>Task description</h3><pre>${escapeHtml(row.task_description)}</pre>` : ''}
    ${row.system_prompt ? `<h3>Trajectory system prompt</h3><pre>${escapeHtml(row.system_prompt)}</pre>` : ''}`;

  $('steps').innerHTML = steps.length
    ? `<section class="panel"><div class="section-heading"><div><h2>Steps</h2><p>Each step shows the message sent to the teacher and the teacher response.</p></div></div>${steps.map((step) => renderStepCard(step, row.system_prompt || '')).join('')}</section>`
    : '<section class="panel empty">No steps recorded.</section>';
}

async function loadPage() {
  if (!trajectoryId) {
    showMessage('Missing trajectory id in URL.', 'error');
    return;
  }

  try {
    const datasetPayload = await ensureDatasetLoaded((text) => showMessage(text, 'info'));
    setServerStatus(`${datasetPayload.summary.totalTrajectories.toLocaleString()} trajectories loaded`);
    const payload = await api(`/api/trajectory?id=${encodeURIComponent(trajectoryId)}`);
    renderTrajectory(payload.trajectory);
    showMessage(`Loaded trajectory ${trajectoryId}.`, 'info');
  } catch (error) {
    setServerStatus('No dataset loaded');
    const hint = error.status === 409 ? ' Go back to Overview and load a directory first.' : '';
    showMessage(`${error.message}${hint}`, 'error');
    $('trajectoryCards').innerHTML = '';
    $('taskPanel').innerHTML = `<div class="empty">${escapeHtml(error.message)}</div>`;
    $('steps').innerHTML = '';
  }
}

loadPage();
