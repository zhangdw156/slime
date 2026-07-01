import assert from 'node:assert/strict';
import { mkdtemp, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { buildDataset, filterSamples, filterTrajectories, getSample, paginate, readSampleTrajectoryRows, readTrajectoryRow } from './data.js';

async function makeDatasetDir() {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'alfworld-viewer-'));
  const rows = [
    {
      trajectory_id: 'train_000001_sample_00',
      sample_id: 1,
      sample_repeat_id: 0,
      split: 'train',
      task_type: 'pick',
      gamefile: 'game1',
      status: 'failed',
      reward: 0,
      done: true,
      invalid_action_count: 1,
      steps: [
        {
          step: 1,
          observation: 'obs failed',
          teacher_response: 'bad',
          user_prompt: 'prompt',
          action: 'look',
          valid_action: false,
          invalid_reason: 'invalid_admissible',
          prompt_tokens: 10,
          response_tokens: 4,
          finish_type: 'stop',
        },
      ],
    },
    {
      trajectory_id: 'train_000001_sample_01',
      sample_id: 1,
      sample_repeat_id: 1,
      split: 'train',
      task_type: 'pick',
      gamefile: 'game1',
      status: 'success',
      reward: 1,
      won: true,
      done: true,
      invalid_action_count: 0,
      steps: [
        {
          step: 1,
          observation: 'obs success',
          teacher_response: 'ok',
          user_prompt: 'prompt',
          action: 'take apple',
          valid_action: true,
          prompt_tokens: 11,
          response_tokens: 3,
          finish_type: 'stop',
        },
      ],
    },
    {
      trajectory_id: 'train_000002_sample_00',
      sample_id: 2,
      sample_repeat_id: 0,
      split: 'train',
      task_type: 'heat',
      gamefile: 'game2',
      status: 'truncated',
      reward: 0,
      truncated: true,
      steps: [],
    },
  ];
  await writeFile(path.join(directory, 'all_trajectories.jsonl'), rows.map((row) => JSON.stringify(row)).join('\n') + '\n');
  return directory;
}

test('buildDataset indexes all trajectories and computes analysis summary', async () => {
  const directory = await makeDatasetDir();
  const dataset = await buildDataset(directory);

  assert.equal(dataset.summary.totalTrajectories, 3);
  assert.equal(dataset.summary.uniqueSamples, 2);
  assert.equal(dataset.summary.statusCounts.success, 1);
  assert.equal(dataset.summary.statusCounts.failed, 1);
  assert.equal(dataset.summary.statusCounts.truncated, 1);
  assert.equal(dataset.summary.successRate, 1 / 3);
  assert.equal(dataset.summary.successAtAny, 1 / 2);
  assert.equal(dataset.summary.selectedSftSamples, 1);
  assert.equal(dataset.summary.invalidReasonCounts.invalid_admissible, 1);
  assert.equal(dataset.summary.taskTypeStats[0].taskType, 'pick');
});

test('filters samples and trajectories, paginates, and reads detail rows by offset', async () => {
  const directory = await makeDatasetDir();
  const dataset = await buildDataset(directory);

  assert.equal(filterSamples(dataset, { status: 'success' }).length, 1);
  assert.equal(getSample(dataset, 1).bestSuccessId, 'train_000001_sample_01');
  assert.equal(filterTrajectories(dataset, { sampleId: 1 }).length, 2);
  assert.equal(filterTrajectories(dataset, { status: 'truncated' })[0].id, 'train_000002_sample_00');
  assert.equal(filterTrajectories(dataset, { q: 'game1' }).length, 2);

  const page = paginate(dataset.trajectories, { page: 2, pageSize: 2 });
  assert.equal(page.page, 2);
  assert.equal(page.items.length, 1);

  const detail = await readTrajectoryRow(dataset, 'train_000001_sample_01');
  assert.equal(detail.steps[0].teacher_response, 'ok');

  const sampleRows = await readSampleTrajectoryRows(dataset, 1);
  assert.deepEqual(sampleRows.map((row) => row.trajectory_id), ['train_000001_sample_00', 'train_000001_sample_01']);
});
