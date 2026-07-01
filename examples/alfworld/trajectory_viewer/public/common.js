export const DIRECTORY_STORAGE_KEY = 'alfworldViewerDirectory';

export const $ = (id) => document.getElementById(id);

export function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

export function formatNumber(value, digits = 0) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '0';
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '0%';
  return `${(number * 100).toFixed(1)}%`;
}

export async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { 'content-type': 'application/json', ...(options.headers || {}) },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error || `HTTP ${response.status}`);
    error.status = response.status;
    error.details = payload.details;
    throw error;
  }
  return payload;
}

export function showMessage(text, type = 'info') {
  const message = $('message');
  if (!message) return;
  if (!text) {
    message.className = 'message hidden';
    message.textContent = '';
    return;
  }
  message.className = `message ${type}`;
  message.textContent = text;
}

export function statusBadge(status) {
  return `<span class="badge ${escapeHtml(status)}">${escapeHtml(status)}</span>`;
}

function softBreakHtml(value) {
  return escapeHtml(value).replaceAll('/', '/<wbr>').replaceAll('_', '_<wbr>').replaceAll('-', '-<wbr>');
}

export function card(label, value, sub = '') {
  const valueText = String(value ?? '');
  const subText = String(sub ?? '');
  const longClass = valueText.length > 18 || subText.length > 48 ? ' long-value' : '';
  return `<article class="card${longClass}"><div class="label">${escapeHtml(label)}</div><div class="value">${softBreakHtml(value)}</div>${sub ? `<div class="sub">${softBreakHtml(sub)}</div>` : ''}</article>`;
}

export function makePager(kind, payload) {
  return `<span>Page ${payload.page} / ${payload.totalPages} · ${formatNumber(payload.total)} rows</span><button data-page-kind="${escapeHtml(kind)}" data-page="${Math.max(1, payload.page - 1)}">Prev</button><button data-page-kind="${escapeHtml(kind)}" data-page="${Math.min(payload.totalPages, payload.page + 1)}">Next</button>`;
}

export function sampleHref(sampleId) {
  return `/sample.html?id=${encodeURIComponent(sampleId)}`;
}

export function trajectoryHref(trajectoryId, sampleId = '') {
  const params = new URLSearchParams({ id: trajectoryId });
  if (sampleId !== '') params.set('sampleId', sampleId);
  return `/trajectory.html?${params}`;
}

export function anchor(href, text, className = '') {
  return `<a class="link ${escapeHtml(className)}" href="${escapeHtml(href)}">${escapeHtml(text)}</a>`;
}

export function currentQuery() {
  return new URLSearchParams(window.location.search);
}

export function setServerStatus(text) {
  const status = $('serverStatus');
  if (status) status.textContent = text;
}

export function rememberDirectory(directory) {
  if (directory) localStorage.setItem(DIRECTORY_STORAGE_KEY, directory);
}

export function rememberedDirectory() {
  return localStorage.getItem(DIRECTORY_STORAGE_KEY) || '';
}

export async function loadDataset(directory) {
  const payload = await api('/api/load', { method: 'POST', body: JSON.stringify({ directory }) });
  rememberDirectory(directory);
  return payload;
}

export async function ensureDatasetLoaded(onStatus = () => {}) {
  try {
    return await api('/api/summary');
  } catch (error) {
    if (error.status !== 409) throw error;
    const directory = rememberedDirectory();
    if (!directory) throw error;
    onStatus(`Loading remembered directory ${directory} ...`);
    return await loadDataset(directory);
  }
}

export function renderTrajectoryTable(items, { includeSample = true } = {}) {
  const columnCount = includeSample ? 8 : 7;
  const header = `<thead><tr><th>Trajectory</th>${includeSample ? '<th>Sample</th>' : ''}<th>Status</th><th>Task</th><th>Steps</th><th>Invalid</th><th>Tokens</th><th>Terminal</th></tr></thead>`;
  if (!items.length) {
    return `${header}<tbody><tr><td colspan="${columnCount}" class="empty">No trajectories match the current filters.</td></tr></tbody>`;
  }
  return `
    ${header}
    <tbody>${items
      .map(
        (item) => `<tr><td>${anchor(trajectoryHref(item.id, item.sampleId), item.id, 'mono')}</td>${includeSample ? `<td class="mono">${anchor(sampleHref(item.sampleId), item.sampleId)}${item.repeatId !== null ? `<div class="muted">repeat ${escapeHtml(item.repeatId)}</div>` : ''}</td>` : ''}<td>${statusBadge(item.status)}</td><td>${escapeHtml(item.taskType)}<div class="muted">${escapeHtml(item.split)}</div></td><td>${formatNumber(item.numSteps)}</td><td>${formatNumber(item.invalidActionCount)}</td><td><span class="muted">p</span>${formatNumber(item.promptTokens)} / <span class="muted">r</span>${formatNumber(item.responseTokens)}</td><td>${escapeHtml(item.terminalReason || item.error || '')}</td></tr>`,
      )
      .join('')}</tbody>`;
}

function textBlock(title, value, { open = false } = {}) {
  const text = value || '';
  const chars = formatNumber(String(text).length);
  return `<details class="text-block" ${open ? 'open' : ''}><summary><h3>${escapeHtml(title)}</h3><span class="muted">${chars} chars</span></summary><pre>${escapeHtml(text)}</pre></details>`;
}

export function renderStepCard(step, inheritedSystemPrompt = '') {
  const valid = step.valid_action !== false && step.valid_format !== false && step.valid_admissible !== false;
  const systemPrompt = step.system_prompt || inheritedSystemPrompt;
  const action = step.action || step.parsed_action || '';
  const optionalBlocks = [
    systemPrompt ? textBlock('System prompt', systemPrompt) : '',
    textBlock('User prompt / message', step.user_prompt || ''),
    textBlock('Admissible actions', Array.isArray(step.admissible_actions) ? step.admissible_actions.join('\n') : ''),
  ].join('');
  return `<article class="step-card">
    <div class="step-header"><div><strong>Step ${escapeHtml(step.step)}</strong> ${valid ? '<span class="badge success">valid</span>' : '<span class="badge failed">invalid</span>'}</div><div class="muted action-label">action: <span class="mono">${escapeHtml(action)}</span></div></div>
    <div class="step-body">
      ${textBlock('Observation', step.observation || '', { open: true })}
      ${textBlock('Teacher response', step.teacher_response || '', { open: true })}
      ${optionalBlocks}
    </div>
    <p class="muted step-meta">reward=${escapeHtml(step.reward ?? 0)} · done=${escapeHtml(step.done)} · won=${escapeHtml(step.won)} · finish=${escapeHtml(step.finish_type || '')} · prompt_tokens=${formatNumber(step.prompt_tokens)} · response_tokens=${formatNumber(step.response_tokens)} ${step.invalid_reason ? `· invalid_reason=${escapeHtml(step.invalid_reason)}` : ''}</p>
  </article>`;
}
