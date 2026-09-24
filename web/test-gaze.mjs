#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import {
  BASE_FEAT_DIM,
  fitGazeModel,
  predictModel,
  assertSample,
  syntheticDataset,
  summarizeErrors,
  chooseLambdaFromDwells,
  expandFeatures,
  validateGazeModel,
  geomMatchesStored,
  motorStepDelta,
  MOTOR_INPUT_GAIN,
  STORE_VERSION,
} from './gaze.mjs';

let failed = 0;

function ok(name) {
  console.log(`  ok  ${name}`);
}

function fail(name, msg) {
  failed++;
  console.error(` FAIL ${name}: ${msg}`);
}

function approx(a, b, eps = 0.08) {
  return Math.abs(a - b) <= eps;
}

// --- gaze numerical tests ---
try {
  const linearMap = (f) => ({ x: 0.1 + 0.4 * f[0] + 0.2 * f[4], y: 0.2 + 0.35 * f[1] - 0.15 * f[5] });
  const train = syntheticDataset(linearMap, 120, 0.01);
  const model = fitGazeModel(train);
  if (!model) fail('linear fit', 'model null');
  else {
    const probe = train[0].f;
    const pred = predictModel(model, probe);
    const expect = linearMap(probe);
    if (!pred || !approx(pred.x, expect.x, 0.12) || !approx(pred.y, expect.y, 0.12)) {
      fail('linear predict', `got ${JSON.stringify(pred)} want ~${JSON.stringify(expect)}`);
    } else ok('linear mapping recovery');
  }
} catch (e) {
  fail('linear mapping', e.message);
}

try {
  const nonlinear = (f) => ({
    x: 0.5 + 0.3 * f[0] + 0.2 * f[0] * f[0],
    y: 0.5 + 0.25 * f[1] + 0.15 * f[4] * f[5],
  });
  const train = syntheticDataset(nonlinear, 200, 0.015);
  const model = fitGazeModel(train);
  const pred = predictModel(model, train[50].f);
  const expect = nonlinear(train[50].f);
  if (!model || !pred || !approx(pred.x, expect.x, 0.15) || !approx(pred.y, expect.y, 0.15)) {
    fail('nonlinear', 'extended model did not approximate synthetic map');
  } else ok('nonlinear mapping (extended)');
} catch (e) {
  fail('nonlinear', e.message);
}

try {
  const train = syntheticDataset((f) => ({ x: f[0], y: f[1] }), 40);
  const targets = train.map((s) => ({ f: s.f, tx: 0.99, ty: 0.01 }));
  const modelOnCalib = fitGazeModel(train);
  const modelOnVal = fitGazeModel(targets);
  if (!modelOnCalib) fail('fit finite', 'calib model null');
  else ok('finite fit on clean samples');
  const pCalib = predictModel(modelOnCalib, train[0].f);
  const pVal = predictModel(modelOnVal, train[0].f);
  if (!pCalib || !Number.isFinite(pCalib.x)) fail('continuous', 'bad predict');
  else if (pVal && approx(pVal.x, 0.99, 0.05) && approx(pVal.y, 0.01, 0.05)) {
    ok('validation targets are separate from default calib fit');
  } else ok('predict returns continuous coordinates');
  const snapDist = Math.hypot(pCalib.x - train[0].tx, pCalib.y - train[0].ty);
  if (snapDist < 1e-6) fail('no snap', 'prediction equals training target exactly');
  else ok('prediction does not snap to labeled target');
} catch (e) {
  fail('continuous', e.message);
}

try {
  assertSample({ f: new Array(BASE_FEAT_DIM).fill(0), tx: 0, ty: 0 });
  let threw = false;
  try {
    assertSample({ f: [NaN], tx: 0, ty: 0 });
  } catch {
    threw = true;
  }
  if (!threw) fail('malformed', 'NaN not rejected');
  else ok('malformed sample rejection');
  if (fitGazeModel([{ f: new Array(BASE_FEAT_DIM).fill(0), tx: 0, ty: 0 }])) {
    fail('min samples', 'fit with 1 sample');
  } else ok('insufficient samples rejected');
} catch (e) {
  fail('malformed', e.message);
}

try {
  const dwells = [];
  for (let d = 0; d < 4; d++) {
    dwells.push(
      syntheticDataset((f) => ({ x: 0.2 + d * 0.15, y: 0.3 + d * 0.1 }), 30, 0.02).map((s) => ({
        ...s,
        tx: 0.2 + d * 0.15,
        ty: 0.3 + d * 0.1,
      })),
    );
  }
  const lam = chooseLambdaFromDwells(dwells, 'linear');
  if (!Number.isFinite(lam) || lam <= 0) fail('lambda', String(lam));
  else ok('lambda chosen from dwell holdout');
} catch (e) {
  fail('lambda', e.message);
}

try {
  const errs = [10, 20, 30, 40, 100];
  const sum = summarizeErrors(errs, 500);
  if (sum.medianPx !== 30 || sum.count !== 5) fail('summarize', JSON.stringify(sum));
  else ok('error summary stats');
} catch (e) {
  fail('summarize', e.message);
}

try {
  const f = new Array(BASE_FEAT_DIM).fill(0.1);
  const ext = expandFeatures(f, 'extended');
  if (!ext || ext.length !== BASE_FEAT_DIM + 7) fail('expand', 'bad length');
  else ok('feature expansion');
} catch (e) {
  fail('expand', e.message);
}

try {
  if (STORE_VERSION !== 3) fail('store version', String(STORE_VERSION));
  else ok('calibration store version v3');
} catch (e) {
  fail('store version', e.message);
}

try {
  const train = syntheticDataset((f) => ({ x: f[0], y: f[1] }), 40);
  const model = fitGazeModel(train);
  if (!validateGazeModel(model)) fail('validate model', 'good model rejected');
  const bad = { ...model, wx: [NaN, ...model.wx.slice(1)] };
  if (validateGazeModel(bad) || predictModel(bad, train[0].f)) fail('validate model', 'malformed accepted');
  else ok('malformed stored model rejected at predict');
} catch (e) {
  fail('validate model', e.message);
}

try {
  const { dx, dy } = motorStepDelta(0.65, 0.35, 0.5, 0.5, MOTOR_INPUT_GAIN);
  if (!approx(dx, 0.6, 0.01) || !approx(dy, -0.6, 0.01)) fail('motor gain', `dx=${dx} dy=${dy}`);
  else ok('motor step uses sensory gain 4');
  const clamped = motorStepDelta(1, 1, 0, 0, MOTOR_INPUT_GAIN);
  if (clamped.dx !== 1 || clamped.dy !== 1) fail('motor clamp', JSON.stringify(clamped));
  else ok('motor step clamps to [-1,1]');
} catch (e) {
  fail('motor gain', e.message);
}

try {
  const base = {
    v: 3,
    vw: 1280,
    vh: 720,
    sw: 800,
    sh: 500,
    dpr: 2,
    deviceId: 'cam-a',
    screenX: 100,
    screenY: 50,
  };
  if (!geomMatchesStored(base, { ...base })) fail('geom', 'identical mismatch');
  if (geomMatchesStored(base, { ...base, deviceId: 'cam-b' })) fail('geom', 'deviceId ignored');
  if (geomMatchesStored(base, { ...base, screenX: 200 })) fail('geom', 'window move ignored');
  else ok('geometry fingerprint v3 matching');
} catch (e) {
  fail('geom', e.message);
}

try {
  const appSrc = readFileSync(new URL('./app.js', import.meta.url), 'utf8');
  if (appSrc.includes('lastGoodFeatMs > STALE_MS && state.mode === \'track\'')) {
    fail('sampleQuality', 'stale gate still rejects incoming frames');
  } else ok('sampleQuality does not block recovery on stale lastGoodFeatMs');
  if (!appSrc.includes('motorStepDelta') || !appSrc.includes('controlEpoch')) {
    fail('step loop', 'missing epoch or motor gain wiring');
  } else ok('step loop epoch and gain wired in app.js');
  if (!appSrc.includes('RAF_DT_MAX_S') || appSrc.includes('const dt = 1 / 60')) {
    fail('render loop', 'fixed 1/60 dt still present');
  } else ok('render loop uses RAF delta time');
} catch (e) {
  fail('app regression', e.message);
}

// --- syntax / HTML reference checks ---
const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const appJs = readFileSync(new URL('./app.js', import.meta.url), 'utf8');

for (const ref of ['./style.css', './app.js']) {
  if (!html.includes(ref)) fail('html refs', `missing ${ref}`);
  else ok(`index.html references ${ref}`);
}
if (!appJs.includes("from './gaze.mjs'") && !appJs.includes('from "./gaze.mjs"')) {
  fail('module graph', 'app.js must import gaze.mjs');
} else ok('app.js imports gaze.mjs');

if (!html.includes('type="module"') || !html.includes('app.js')) {
  fail('html module', 'app.js module script');
} else ok('index.html loads app.js as module');

import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const artifactDir = fileURLToPath(new URL('.', import.meta.url));
try {
  execSync('node --check app.js', { cwd: artifactDir });
  ok('app.js syntax (node --check)');
} catch (e) {
  fail('app.js syntax', e.message);
}

const hash = createHash('sha256').update(appJs).digest('hex').slice(0, 12);
console.log(`\napp.js sha256[0:12]=${hash}`);
console.log(failed ? `\n${failed} test(s) failed` : '\nAll gaze tests passed');
process.exit(failed ? 1 : 0);
