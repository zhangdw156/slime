import { createReadStream } from 'node:fs';
import { open, readdir, realpath, stat } from 'node:fs/promises';
import path from 'node:path';

export const DEFAULT_FILE_NAME = 'all_trajectories.jsonl';
export const DEFAULT_PAGE_SIZE = 50;
export const MAX_PAGE_SIZE = 200;

function toNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function stableString(value, fallback = 'unknown') {
  if (value === null || value === undefined || value === '') {
    return fallback;
  }
  return String(value);
}

function normalizeStatus(row) {
  if (row.status) return String(row.status);
  if (row.error) return 'error';
  if (row.aborted) return 'aborted';
  if (row.reward === 1 || row.reward === 1.0 || row.won === true) return 'success';
  if (row.done) return 'failed';
  if (row.truncated) return 'truncated';
  return 'unknown';
}

function trajectoryIsSuccessful(row) {
  return normalizeStatus(row) === 'success' || toNumber(row.reward) === 1 || row.won === true;
}

function increment(map, key, by = 1) {
  const normalized = stableString(key);
  map.set(normalized, (map.get(normalized) ?? 0) + by);
}

function objectFromMap(map) {
  return Object.fromEntries([...map.entries()].sort(([a], [b]) => a.localeCompare(b)));
}

function summarizeSteps(steps) {
  let promptTokens = 0;
  let responseTokens = 0;
  const finishTypes = new Map();
  const invalidReasons = new Map();
  let invalidStepCount = 0;

  if (!Array.isArray(steps)) {
    return { promptTokens, responseTokens, finishTypes, invalidReasons, invalidStepCount };
  }

  for (const step of steps) {
    promptTokens += toNumber(step?.prompt_tokens);
    responseTokens += toNumber(step?.response_tokens);
    if (step?.finish_type) increment(finishTypes, step.finish_type);
    if (step?.valid_action === false || step?.valid_format === false || step?.valid_admissible === false) {
      invalidStepCount += 1;
      increment(invalidReasons, step?.invalid_reason || 'invalid_action');
    }
  }

  return { promptTokens, responseTokens, finishTypes, invalidReasons, invalidStepCount };
}

function makeTrajectorySummary(row, offset, byteLength, lineNumber) {
  const steps = Array.isArray(row.steps) ? row.steps : [];
  const stepStats = summarizeSteps(steps);
  const status = normalizeStatus(row);
  const sampleId = stableString(row.sample_id ?? row.task_index ?? row.gamefile ?? row.trajectory_id, 'unknown');
  const repeatId = row.sample_repeat_id ?? row.attempt_id ?? null;
  const numSteps = toNumber(row.num_steps, steps.length);
  const invalidActionCount = toNumber(row.invalid_action_count, stepStats.invalidStepCount);

  return {
    id: stableString(row.trajectory_id ?? row.episode_id ?? `line-${lineNumber}`),
    lineNumber,
    offset,
    byteLength,
    sampleId,
    repeatId,
    split: stableString(row.split, 'unknown'),
    taskType: stableString(row.task_type, 'unknown'),
    gamefile: stableString(row.gamefile, ''),
    status,
    terminalReason: stableString(row.terminal_reason, ''),
    reward: toNumber(row.reward),
    won: row.won === true || status === 'success',
    done: row.done === true,
    aborted: row.aborted === true,
    truncated: row.truncated === true,
    error: row.error ? String(row.error) : '',
    numSteps,
    invalidActionCount,
    promptTokens: stepStats.promptTokens,
    responseTokens: stepStats.responseTokens,
    elapsedSec: toNumber(row.elapsed_sec),
    teacherModel: stableString(row.teacher_model, ''),
    finishTypes: objectFromMap(stepStats.finishTypes),
    invalidReasons: objectFromMap(stepStats.invalidReasons),
  };
}

function betterSftCandidate(a, b) {
  if (!a) return b;
  if (b.numSteps !== a.numSteps) return b.numSteps < a.numSteps ? b : a;
  if (b.responseTokens !== a.responseTokens) return b.responseTokens < a.responseTokens ? b : a;
  const aRepeat = toNumber(a.repeatId, 0);
  const bRepeat = toNumber(b.repeatId, 0);
  if (bRepeat !== aRepeat) return bRepeat < aRepeat ? b : a;
  return b.id.localeCompare(a.id) < 0 ? b : a;
}

function makeSampleSummary(sampleId, trajectories) {
  const statusCounts = new Map();
  let bestSuccess = null;
  let taskType = 'unknown';
  let split = 'unknown';
  let gamefile = '';
  let invalidActionCount = 0;
  let totalSteps = 0;

  for (const item of trajectories) {
    increment(statusCounts, item.status);
    invalidActionCount += item.invalidActionCount;
    totalSteps += item.numSteps;
    if (item.taskType !== 'unknown') taskType = item.taskType;
    if (item.split !== 'unknown') split = item.split;
    if (item.gamefile) gamefile = item.gamefile;
    if (item.status === 'success') bestSuccess = betterSftCandidate(bestSuccess, item);
  }

  return {
    sampleId,
    taskType,
    split,
    gamefile,
    attempts: trajectories.length,
    successes: statusCounts.get('success') ?? 0,
    statusCounts: objectFromMap(statusCounts),
    success: (statusCounts.get('success') ?? 0) > 0,
    bestSuccessId: bestSuccess?.id ?? '',
    bestSuccessSteps: bestSuccess?.numSteps ?? null,
    invalidActionCount,
    avgSteps: trajectories.length ? totalSteps / trajectories.length : 0,
    trajectoryIds: trajectories.map((item) => item.id),
  };
}

function avg(total, count) {
  return count ? total / count : 0;
}

function finalizeDataset({ directory, filePath, fileStat, trajectories, parseErrors, startedAt }) {
  const byId = new Map();
  const samples = new Map();
  const statusCounts = new Map();
  const splitCounts = new Map();
  const finishTypeCounts = new Map();
  const invalidReasonCounts = new Map();
  const taskTypeStats = new Map();
  const stepsByStatus = new Map();
  const tokensByStatus = new Map();
  let totalSteps = 0;
  let totalInvalidActions = 0;
  let totalPromptTokens = 0;
  let totalResponseTokens = 0;

  for (const item of trajectories) {
    byId.set(item.id, item);
    if (!samples.has(item.sampleId)) samples.set(item.sampleId, []);
    samples.get(item.sampleId).push(item);

    increment(statusCounts, item.status);
    increment(splitCounts, item.split);
    totalSteps += item.numSteps;
    totalInvalidActions += item.invalidActionCount;
    totalPromptTokens += item.promptTokens;
    totalResponseTokens += item.responseTokens;

    for (const [key, value] of Object.entries(item.finishTypes)) increment(finishTypeCounts, key, value);
    for (const [key, value] of Object.entries(item.invalidReasons)) increment(invalidReasonCounts, key, value);

    if (!taskTypeStats.has(item.taskType)) {
      taskTypeStats.set(item.taskType, { taskType: item.taskType, trajectories: 0, successes: 0, samples: new Set() });
    }
    const taskStats = taskTypeStats.get(item.taskType);
    taskStats.trajectories += 1;
    taskStats.samples.add(item.sampleId);
    if (item.status === 'success') taskStats.successes += 1;

    if (!stepsByStatus.has(item.status)) stepsByStatus.set(item.status, { total: 0, count: 0 });
    stepsByStatus.get(item.status).total += item.numSteps;
    stepsByStatus.get(item.status).count += 1;

    if (!tokensByStatus.has(item.status)) tokensByStatus.set(item.status, { prompt: 0, response: 0, count: 0 });
    tokensByStatus.get(item.status).prompt += item.promptTokens;
    tokensByStatus.get(item.status).response += item.responseTokens;
    tokensByStatus.get(item.status).count += 1;
  }

  const sampleSummaries = [...samples.entries()].map(([sampleId, values]) => makeSampleSummary(sampleId, values));
  sampleSummaries.sort((a, b) => a.sampleId.localeCompare(b.sampleId, undefined, { numeric: true }));
  const selectedSftCandidates = sampleSummaries
    .filter((item) => item.bestSuccessId)
    .map((item) => byId.get(item.bestSuccessId))
    .filter(Boolean);
  selectedSftCandidates.sort((a, b) => a.sampleId.localeCompare(b.sampleId, undefined, { numeric: true }));

  const completeSamples = sampleSummaries.filter((item) => item.attempts >= 8).length;
  const successfulSamples = sampleSummaries.filter((item) => item.success).length;
  const completedSuccessfulSamples = sampleSummaries.filter((item) => item.attempts >= 8 && item.success).length;

  const taskTypeSummary = [...taskTypeStats.values()]
    .map((item) => ({
      taskType: item.taskType,
      trajectories: item.trajectories,
      successes: item.successes,
      samples: item.samples.size,
      successRate: avg(item.successes, item.trajectories),
    }))
    .sort((a, b) => b.trajectories - a.trajectories || a.taskType.localeCompare(b.taskType));

  const avgStepsByStatus = Object.fromEntries(
    [...stepsByStatus.entries()].map(([status, item]) => [status, avg(item.total, item.count)]).sort(([a], [b]) => a.localeCompare(b)),
  );
  const avgTokensByStatus = Object.fromEntries(
    [...tokensByStatus.entries()]
      .map(([status, item]) => [status, { promptTokens: avg(item.prompt, item.count), responseTokens: avg(item.response, item.count) }])
      .sort(([a], [b]) => a.localeCompare(b)),
  );

  const summary = {
    directory,
    filePath,
    fileName: path.basename(filePath),
    fileSizeBytes: fileStat.size,
    fileModifiedAt: fileStat.mtime.toISOString(),
    indexedAt: new Date().toISOString(),
    indexDurationMs: Date.now() - startedAt,
    totalTrajectories: trajectories.length,
    uniqueSamples: sampleSummaries.length,
    statusCounts: objectFromMap(statusCounts),
    splitCounts: objectFromMap(splitCounts),
    successRate: avg(statusCounts.get('success') ?? 0, trajectories.length),
    successAtAny: avg(successfulSamples, sampleSummaries.length),
    completeSamples,
    successAt8Complete: avg(completedSuccessfulSamples, completeSamples),
    selectedSftSamples: selectedSftCandidates.length,
    parseErrorCount: parseErrors.length,
    parseErrors: parseErrors.slice(0, 20),
    avgSteps: avg(totalSteps, trajectories.length),
    avgInvalidActions: avg(totalInvalidActions, trajectories.length),
    avgPromptTokens: avg(totalPromptTokens, trajectories.length),
    avgResponseTokens: avg(totalResponseTokens, trajectories.length),
    avgStepsByStatus,
    avgTokensByStatus,
    finishTypeCounts: objectFromMap(finishTypeCounts),
    invalidReasonCounts: objectFromMap(invalidReasonCounts),
    taskTypeStats: taskTypeSummary,
  };

  return {
    directory,
    filePath,
    summary,
    trajectories,
    byId,
    samples: sampleSummaries,
    selectedSftCandidates,
  };
}

export async function scanJsonl(filePath, onRow) {
  const stream = createReadStream(filePath);
  let carry = Buffer.alloc(0);
  let carryOffset = 0;
  let absoluteOffset = 0;
  let lineNumber = 0;

  for await (const chunk of stream) {
    const buffer = carry.length ? Buffer.concat([carry, chunk]) : chunk;
    const baseOffset = carry.length ? carryOffset : absoluteOffset;
    let start = 0;
    let newlineIndex = buffer.indexOf(0x0a, start);

    while (newlineIndex !== -1) {
      lineNumber += 1;
      const line = buffer.subarray(start, newlineIndex);
      await onRow(line, baseOffset + start, line.length, lineNumber);
      start = newlineIndex + 1;
      newlineIndex = buffer.indexOf(0x0a, start);
    }

    carry = buffer.subarray(start);
    carryOffset = baseOffset + start;
    absoluteOffset += chunk.length;
  }

  if (carry.length) {
    lineNumber += 1;
    await onRow(carry, carryOffset, carry.length, lineNumber);
  }
}

export async function buildDataset(directory, options = {}) {
  const startedAt = Date.now();
  const resolvedDirectory = await realpath(path.resolve(directory));
  const fileName = options.fileName || DEFAULT_FILE_NAME;
  const filePath = path.join(resolvedDirectory, fileName);
  const fileStat = await stat(filePath);
  if (!fileStat.isFile()) {
    throw new Error(`${filePath} is not a file`);
  }

  const trajectories = [];
  const parseErrors = [];
  await scanJsonl(filePath, async (lineBuffer, offset, byteLength, lineNumber) => {
    const text = lineBuffer.toString('utf8').trim();
    if (!text) return;
    try {
      const row = JSON.parse(text);
      trajectories.push(makeTrajectorySummary(row, offset, byteLength, lineNumber));
    } catch (error) {
      parseErrors.push({ lineNumber, offset, error: error.message });
    }
  });

  return finalizeDataset({ directory: resolvedDirectory, filePath, fileStat, trajectories, parseErrors, startedAt });
}

export async function readTrajectoryRow(dataset, trajectoryId) {
  const entry = dataset.byId.get(trajectoryId);
  if (!entry) return null;
  const handle = await open(dataset.filePath, 'r');
  try {
    const buffer = Buffer.alloc(entry.byteLength);
    await handle.read(buffer, 0, entry.byteLength, entry.offset);
    return JSON.parse(buffer.toString('utf8').trim());
  } finally {
    await handle.close();
  }
}

export async function readSampleTrajectoryRows(dataset, sampleId) {
  const summaries = filterTrajectories(dataset, { sampleId });
  const rows = await Promise.all(summaries.map((summary) => readTrajectoryRow(dataset, summary.id)));
  return rows.filter(Boolean);
}

export function paginate(items, query = {}) {
  const pageSize = Math.min(Math.max(Number.parseInt(query.pageSize ?? DEFAULT_PAGE_SIZE, 10) || DEFAULT_PAGE_SIZE, 1), MAX_PAGE_SIZE);
  const page = Math.max(Number.parseInt(query.page ?? 1, 10) || 1, 1);
  const total = items.length;
  const start = (page - 1) * pageSize;
  return {
    page,
    pageSize,
    total,
    totalPages: Math.max(Math.ceil(total / pageSize), 1),
    items: items.slice(start, start + pageSize),
  };
}

export function filterTrajectories(dataset, query = {}) {
  const status = query.status && query.status !== 'all' ? String(query.status) : '';
  const taskType = query.taskType && query.taskType !== 'all' ? String(query.taskType) : '';
  const split = query.split && query.split !== 'all' ? String(query.split) : '';
  const sampleId = query.sampleId ? String(query.sampleId) : '';
  const search = query.q ? String(query.q).toLowerCase() : '';

  return dataset.trajectories.filter((item) => {
    if (status && item.status !== status) return false;
    if (taskType && item.taskType !== taskType) return false;
    if (split && item.split !== split) return false;
    if (sampleId && item.sampleId !== sampleId) return false;
    if (search) {
      const haystack = `${item.id} ${item.sampleId} ${item.gamefile} ${item.taskType} ${item.status}`.toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });
}

export function getSample(dataset, sampleId) {
  return dataset.samples.find((item) => item.sampleId === String(sampleId)) || null;
}

export function filterSamples(dataset, query = {}) {
  const status = query.status && query.status !== 'all' ? String(query.status) : '';
  const taskType = query.taskType && query.taskType !== 'all' ? String(query.taskType) : '';
  const split = query.split && query.split !== 'all' ? String(query.split) : '';
  const search = query.q ? String(query.q).toLowerCase() : '';

  return dataset.samples.filter((item) => {
    if (status === 'success' && !item.success) return false;
    if (status && status !== 'success' && !item.statusCounts[status]) return false;
    if (taskType && item.taskType !== taskType) return false;
    if (split && item.split !== split) return false;
    if (search) {
      const haystack = `${item.sampleId} ${item.gamefile} ${item.taskType} ${item.bestSuccessId}`.toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  });
}

export function datasetOptions(dataset) {
  const statuses = new Set();
  const taskTypes = new Set();
  const splits = new Set();
  for (const item of dataset.trajectories) {
    statuses.add(item.status);
    taskTypes.add(item.taskType);
    splits.add(item.split);
  }
  return {
    statuses: [...statuses].sort(),
    taskTypes: [...taskTypes].sort(),
    splits: [...splits].sort(),
  };
}

export async function browseDirectory(directory = process.cwd()) {
  const resolvedDirectory = await realpath(path.resolve(directory));
  const entries = await readdir(resolvedDirectory, { withFileTypes: true });
  const children = entries
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .filter((name) => !name.startsWith('.'))
    .sort((a, b) => a.localeCompare(b));
  const files = new Set(entries.filter((entry) => entry.isFile()).map((entry) => entry.name));
  return {
    directory: resolvedDirectory,
    parent: path.dirname(resolvedDirectory),
    hasTrajectoryFile: files.has(DEFAULT_FILE_NAME),
    children,
  };
}
