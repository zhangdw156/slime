import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  browseDirectory,
  buildDataset,
  datasetOptions,
  filterSamples,
  filterTrajectories,
  getSample,
  paginate,
  readSampleTrajectoryRows,
  readTrajectoryRow,
} from './data.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.resolve(__dirname, '..', 'public');
const host = process.env.ALFWORLD_VIEWER_HOST || '127.0.0.1';
const port = Number.parseInt(process.env.ALFWORLD_VIEWER_PORT || '5173', 10);
let currentDataset = null;

const contentTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
};

function sendJson(res, statusCode, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
    'cache-control': 'no-store',
  });
  res.end(body);
}

function sendError(res, statusCode, message, details = undefined) {
  sendJson(res, statusCode, { error: message, details });
}

async function readBodyJson(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (!chunks.length) return {};
  const text = Buffer.concat(chunks).toString('utf8');
  return text ? JSON.parse(text) : {};
}

function requireDataset(res) {
  if (!currentDataset) {
    sendError(res, 409, 'No trajectory directory loaded yet. Choose a directory containing all_trajectories.jsonl first.');
    return null;
  }
  return currentDataset;
}

async function serveStatic(req, res, pathname) {
  const relative = pathname === '/' ? 'index.html' : decodeURIComponent(pathname.slice(1));
  const filePath = path.resolve(publicDir, relative);
  if (filePath !== publicDir && !filePath.startsWith(`${publicDir}${path.sep}`)) {
    sendError(res, 403, 'Forbidden');
    return;
  }

  try {
    const fileStat = await stat(filePath);
    if (!fileStat.isFile()) {
      sendError(res, 404, 'Not found');
      return;
    }
    const ext = path.extname(filePath);
    res.writeHead(200, {
      'content-type': contentTypes[ext] || 'application/octet-stream',
      'content-length': fileStat.size,
      'cache-control': 'no-store',
    });
    createReadStream(filePath).pipe(res);
  } catch {
    sendError(res, 404, 'Not found');
  }
}

async function handleApi(req, res, url) {
  if (req.method === 'GET' && url.pathname === '/api/health') {
    sendJson(res, 200, { ok: true, loaded: Boolean(currentDataset), directory: currentDataset?.directory || null });
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/browse') {
    const directory = url.searchParams.get('dir') || process.env.ALFWORLD_TRAJECTORY_DIR || process.cwd();
    sendJson(res, 200, await browseDirectory(directory));
    return;
  }

  if (req.method === 'POST' && url.pathname === '/api/load') {
    const body = await readBodyJson(req);
    if (!body.directory) {
      sendError(res, 400, 'Missing directory');
      return;
    }
    currentDataset = await buildDataset(body.directory);
    sendJson(res, 200, {
      summary: currentDataset.summary,
      options: datasetOptions(currentDataset),
    });
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/summary') {
    const dataset = requireDataset(res);
    if (!dataset) return;
    sendJson(res, 200, { summary: dataset.summary, options: datasetOptions(dataset) });
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/trajectories') {
    const dataset = requireDataset(res);
    if (!dataset) return;
    const filtered = filterTrajectories(dataset, Object.fromEntries(url.searchParams.entries()));
    sendJson(res, 200, paginate(filtered, Object.fromEntries(url.searchParams.entries())));
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/samples') {
    const dataset = requireDataset(res);
    if (!dataset) return;
    const filtered = filterSamples(dataset, Object.fromEntries(url.searchParams.entries()));
    sendJson(res, 200, paginate(filtered, Object.fromEntries(url.searchParams.entries())));
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/sample') {
    const dataset = requireDataset(res);
    if (!dataset) return;
    const sampleId = url.searchParams.get('id');
    if (!sampleId) {
      sendError(res, 400, 'Missing sample id');
      return;
    }
    const sample = getSample(dataset, sampleId);
    if (!sample) {
      sendError(res, 404, `Unknown sample id: ${sampleId}`);
      return;
    }
    const trajectories = filterTrajectories(dataset, { sampleId });
    sendJson(res, 200, { sample, trajectories });
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/sample-detail') {
    const dataset = requireDataset(res);
    if (!dataset) return;
    const sampleId = url.searchParams.get('id');
    if (!sampleId) {
      sendError(res, 400, 'Missing sample id');
      return;
    }
    const sample = getSample(dataset, sampleId);
    if (!sample) {
      sendError(res, 404, `Unknown sample id: ${sampleId}`);
      return;
    }
    const trajectories = filterTrajectories(dataset, { sampleId });
    const rows = await readSampleTrajectoryRows(dataset, sampleId);
    sendJson(res, 200, { sample, trajectories, rows });
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/sft-candidates') {
    const dataset = requireDataset(res);
    if (!dataset) return;
    sendJson(res, 200, paginate(dataset.selectedSftCandidates, Object.fromEntries(url.searchParams.entries())));
    return;
  }

  if (req.method === 'GET' && url.pathname === '/api/trajectory') {
    const dataset = requireDataset(res);
    if (!dataset) return;
    const id = url.searchParams.get('id');
    if (!id) {
      sendError(res, 400, 'Missing trajectory id');
      return;
    }
    const row = await readTrajectoryRow(dataset, id);
    if (!row) {
      sendError(res, 404, `Unknown trajectory id: ${id}`);
      return;
    }
    sendJson(res, 200, { trajectory: row });
    return;
  }

  sendError(res, 404, 'Unknown API route');
}

export async function requestListener(req, res) {
  const url = new URL(req.url || '/', `http://${req.headers.host || `${host}:${port}`}`);
  try {
    if (url.pathname.startsWith('/api/')) {
      await handleApi(req, res, url);
    } else {
      await serveStatic(req, res, url.pathname);
    }
  } catch (error) {
    sendError(res, 500, error.message, process.env.NODE_ENV === 'production' ? undefined : error.stack);
  }
}

if (fileURLToPath(import.meta.url) === path.resolve(process.argv[1] || '')) {
  const server = http.createServer(requestListener);
  server.listen(port, host, () => {
    console.log(`ALFWorld trajectory viewer: http://${host}:${port}`);
    console.log('Open the page, choose a directory containing all_trajectories.jsonl, then click Load.');
  });
}
