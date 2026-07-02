import assert from 'node:assert/strict';
import test from 'node:test';

class FakeClassList {
  constructor(element) {
    this.element = element;
  }

  add(...classes) {
    const current = new Set(String(this.element.className || '').split(/\s+/).filter(Boolean));
    for (const item of classes) current.add(item);
    this.element.className = [...current].join(' ');
  }

  remove(...classes) {
    const removeSet = new Set(classes);
    this.element.className = String(this.element.className || '')
      .split(/\s+/)
      .filter((item) => item && !removeSet.has(item))
      .join(' ');
  }
}

class FakeElement {
  constructor(id) {
    this.id = id;
    this.textContent = '';
    this.innerHTML = '';
    this.className = '';
    this.classList = new FakeClassList(this);
    this.lastMatches = new Map();
  }

  querySelector() {
    return null;
  }

  querySelectorAll(selector) {
    const matches = [];
    const attr = selector.match(/^\[data-([\w-]+)\]$/)?.[1];
    if (attr) {
      const camel = attr.replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
      const pattern = new RegExp(`data-${attr}="([^"]+)"`, 'g');
      for (const match of this.innerHTML.matchAll(pattern)) {
        matches.push(new FakeInteractiveElement(camel, match[1]));
      }
    }
    this.lastMatches.set(selector, matches);
    return matches;
  }
}

class FakeInteractiveElement {
  constructor(datasetKey, value) {
    this.dataset = { [datasetKey]: value };
    this.listeners = new Map();
  }

  addEventListener(event, callback) {
    this.listeners.set(event, callback);
  }

  click() {
    this.listeners.get('click')?.();
  }
}

function jsonResponse(payload, ok = true, status = 200) {
  return {
    ok,
    status,
    async json() {
      return payload;
    },
  };
}

test('sample page renders an action graph without runtime graph scope errors', async () => {
  const elements = new Map();
  const ids = ['sampleSubtitle', 'sampleCards', 'attemptsTable', 'message', 'serverStatus', 'treeSummary', 'trajectoryTree', 'stateTreeSummary', 'stateTransitionTree'];
  for (const id of ids) elements.set(id, new FakeElement(id));

  globalThis.document = {
    getElementById(id) {
      return elements.get(id) || null;
    },
  };
  globalThis.window = { location: { search: '?id=sample-1' } };
  globalThis.localStorage = { getItem: () => '', setItem: () => {} };
  globalThis.fetch = async (path) => {
    if (path === '/api/summary') {
      return jsonResponse({ summary: { totalTrajectories: 1 } });
    }
    if (String(path).startsWith('/api/sample-detail')) {
      const row = {
        trajectory_id: 'sample_00',
        sample_repeat_id: 0,
        status: 'success',
        steps: [{ action: 'look' }, { action: 'open drawer 1' }],
      };
      return jsonResponse({
        sample: {
          sampleId: 'sample-1',
          taskType: 'pick',
          split: 'train',
          attempts: 1,
          successes: 1,
          statusCounts: { success: 1 },
          success: true,
          bestSuccessId: 'sample_00',
          bestSuccessSteps: 2,
          invalidActionCount: 0,
          avgSteps: 2,
          gamefile: '/data/private/alfworld/game.tw-pddl',
        },
        trajectories: [{ id: 'sample_00', sampleId: 'sample-1', repeatId: 0, status: 'success', taskType: 'pick', split: 'train', numSteps: 2, invalidActionCount: 0, promptTokens: 0, responseTokens: 0, terminalReason: '' }],
        rows: [row],
      });
    }
    return jsonResponse({ error: `unexpected ${path}` }, false, 404);
  };

  await import(`./sample.js?runtime-test=${Date.now()}`);
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));

  assert.doesNotMatch(elements.get('message').textContent, /graph is not defined/);
  assert.match(elements.get('trajectoryTree').innerHTML, /graph-canvas/);
  assert.match(elements.get('stateTransitionTree').innerHTML, /State branch points|state-edge-label|open drawer 1/);
  assert.doesNotMatch(elements.get('sampleCards').innerHTML, /game\.tw-pddl|Gamefile|private/);
});

test('highlighting a trajectory shows order on nodes without edge step labels', async () => {
  const elements = new Map();
  const ids = ['sampleSubtitle', 'sampleCards', 'attemptsTable', 'message', 'serverStatus', 'treeSummary', 'trajectoryTree', 'stateTreeSummary', 'stateTransitionTree'];
  for (const id of ids) elements.set(id, new FakeElement(id));

  globalThis.document = {
    getElementById(id) {
      return elements.get(id) || null;
    },
  };
  globalThis.window = { location: { search: '?id=sample-2' } };
  globalThis.localStorage = { getItem: () => '', setItem: () => {} };
  globalThis.fetch = async (path) => {
    if (path === '/api/summary') return jsonResponse({ summary: { totalTrajectories: 1 } });
    if (String(path).startsWith('/api/sample-detail')) {
      const row = {
        trajectory_id: 'sample_01',
        sample_repeat_id: 0,
        status: 'success',
        steps: [{ action: 'go to desk 1' }, { action: 'go to bed 1' }, { action: 'go to desk 1' }],
      };
      return jsonResponse({
        sample: {
          sampleId: 'sample-2',
          taskType: 'pick',
          split: 'train',
          attempts: 1,
          successes: 1,
          statusCounts: { success: 1 },
          success: true,
          bestSuccessId: 'sample_01',
          bestSuccessSteps: 3,
          invalidActionCount: 0,
          avgSteps: 3,
        },
        trajectories: [{ id: 'sample_01', sampleId: 'sample-2', repeatId: 0, status: 'success', taskType: 'pick', split: 'train', numSteps: 3, invalidActionCount: 0, promptTokens: 0, responseTokens: 0, terminalReason: '' }],
        rows: [row],
      });
    }
    return jsonResponse({ error: `unexpected ${path}` }, false, 404);
  };

  await import(`./sample.js?highlight-test=${Date.now()}`);
  await new Promise((resolve) => setImmediate(resolve));
  await new Promise((resolve) => setImmediate(resolve));

  const trajectoryButtons = elements.get('trajectoryTree').lastMatches.get('[data-trajectory-id]');
  assert.equal(trajectoryButtons.length, 1);
  trajectoryButtons[0].click();
  await new Promise((resolve) => setImmediate(resolve));

  const treeHtml = elements.get('trajectoryTree').innerHTML;
  assert.match(treeHtml, /selected-sequence/);
  assert.match(treeHtml, /graph-node-order/);
  assert.match(treeHtml, /graph-node-order[\s\S]*1,3/);
  assert.doesNotMatch(treeHtml, /trajectory-step-label/);
});
