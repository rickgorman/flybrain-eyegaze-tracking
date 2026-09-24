import { FaceLandmarker, FilesetResolver } from './vendor/vision_bundle.mjs';
import {
  BASE_FEAT_DIM,
  STORE_VERSION,
  fitGazeModel,
  predictModel,
  summarizeErrors,
  validateGazeModel,
  geomMatchesStored,
  motorStepDelta,
  MOTOR_INPUT_GAIN,
} from './gaze.mjs';
import { accuracyProtocolAllowed, cameraSessionOwned } from './protocol-guards.mjs';

const LEFT = { o: 33, i: 133, t: 159, b: 145, c: 468 };
const RIGHT = { o: 263, i: 362, t: 386, b: 374, c: 473 };

const STORE_KEY = 'fly-gaze.calibration.v3';
const STALE_MS = 280;
const STEP_HZ = 8;
const STEP_TIMEOUT_MS = 2000;
const STEP_RESPONSE_MAX_MS = 300;
const MOTOR_CMD_MAX_MS = 300;
const RAF_DT_MAX_S = 0.05;
const FLY_SPEED = 0.5;
const SETTLE_MS = 350;
const COLLECT_MIN_MS = 600;
const COLLECT_MAX_MS = 900;
const MIN_DWELL_SAMPLES = 10;
const MAX_DWELL_FAILS = 4;

const GRID_BASE = [
  [0.5, 0.5],
  [0.1, 0.12], [0.5, 0.1], [0.9, 0.12],
  [0.12, 0.5], [0.88, 0.5],
  [0.1, 0.88], [0.5, 0.9], [0.9, 0.88],
  [0.28, 0.28], [0.72, 0.28], [0.28, 0.72], [0.72, 0.72],
  [0.2, 0.5], [0.8, 0.5], [0.5, 0.25], [0.5, 0.75],
];

const HOLDOUT_NORM = [
  [0.18, 0.38],
  [0.82, 0.42],
  [0.35, 0.62],
  [0.65, 0.58],
  [0.5, 0.35],
  [0.42, 0.82],
];

const $ = (id) => document.getElementById(id);
const cam = $('cam');
const stage = $('stage');
const pip = $('pip');
const overlay = $('overlay');
const statusEl = $('status');
const telemetryEl = $('telemetry');
const introEl = $('intro');

const ctx = stage.getContext('2d');
const pctx = pip.getContext('2d');

class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.4) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
  filter(x, t) {
    if (this.tPrev == null) {
      this.tPrev = t;
      this.xPrev = x;
      return x;
    }
    const dt = Math.max(1e-3, t - this.tPrev);
    const dx = (x - this.xPrev) / dt;
    const edx = lp(dx, this.dxPrev, alpha(1, dt));
    this.dxPrev = edx;
    const y = lp(x, this.xPrev, alpha(this.minCutoff + this.beta * Math.abs(edx), dt));
    this.xPrev = y;
    this.tPrev = t;
    return y;
  }
  reset() {
    this.xPrev = null;
    this.dxPrev = 0;
    this.tPrev = null;
  }
}

function alpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}
function lp(x, prev, a) {
  return a * x + (1 - a) * prev;
}

const state = {
  landmarker: null,
  stream: null,
  loopsOn: false,
  inferScheduled: false,
  rafId: 0,
  lastVideoTs: 0,
  lastGoodFeatMs: 0,
  face: false,
  feats: null,
  blink: false,
  model: null,
  dwellBuckets: [],
  heldout: null,
  mode: 'idle',
  calib: null,
  accuracy: null,
  smoothX: new OneEuro(1.15, 0.42),
  smoothY: new OneEuro(1.15, 0.42),
  gazeNorm: null,
  gazePx: null,
  uncertaintyPx: null,
  fly: { x: 0.5, y: 0.5, vx: 0, vy: 0 },
  stageCss: { w: 1, h: 1, dpr: 1 },
  geom: null,
  geomStale: false,
  headRef: null,
  pointer: false,
  pointerNorm: null,
  paused: false,
  backend: { ready: false, training: false, progress: 0, error: null, model: null, latencyMs: null },
  stepSeq: 0,
  lastStepSent: 0,
  lastStepAck: -1,
  pollTimer: 0,
  stats: { inferMs: 0, inferFps: 0, drawFps: 0, delegate: '—' },
  controlEpoch: 0,
  stepInFlight: false,
  lastMotorCmdMs: 0,
  lastRafTs: 0,
  cameraGen: 0,
  camStarting: false,
  stopFrozen: false,
  prevFace: false,
  prevBlink: false,
};

function setStatus(text, kind = '') {
  statusEl.textContent = text;
  statusEl.className = kind;
}

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function buildCalibSequence() {
  const pass1 = shuffle(GRID_BASE.filter((p) => p[0] !== 0.5 || p[1] !== 0.5));
  const pass2 = shuffle(GRID_BASE);
  return pass1.concat(pass2);
}

function blendMap(blends) {
  const m = Object.create(null);
  for (const c of blends?.categories || []) m[c.categoryName] = c.score;
  return m;
}

function euler(mat) {
  const d = mat?.data;
  if (!d || d.length < 16) return { yaw: 0, pitch: 0, roll: 0 };
  const pitch = Math.asin(Math.max(-1, Math.min(1, -d[6])));
  const yaw = Math.atan2(d[2], d[10]);
  const roll = Math.atan2(d[4], d[5]);
  return { yaw, pitch, roll };
}

function ear(lm, e, vw, vh) {
  const tx = lm[e.t].x * vw;
  const ty = lm[e.t].y * vh;
  const bx = lm[e.b].x * vw;
  const by = lm[e.b].y * vh;
  const ox = lm[e.o].x * vw;
  const oy = lm[e.o].y * vh;
  const ix = lm[e.i].x * vw;
  const iy = lm[e.i].y * vh;
  const v = Math.hypot(tx - bx, ty - by);
  const h = Math.hypot(ox - ix, oy - iy);
  return h > 1e-5 ? v / h : 0.25;
}

function irisLocal(lm, e, vw, vh) {
  const o = { x: lm[e.o].x * vw, y: lm[e.o].y * vh };
  const i = { x: lm[e.i].x * vw, y: lm[e.i].y * vh };
  const c = { x: lm[e.c].x * vw, y: lm[e.c].y * vh };
  const ex = i.x - o.x;
  const ey = i.y - o.y;
  const w = Math.hypot(ex, ey);
  if (w < 1e-5) return { along: 0.5, perp: 0 };
  const ux = ex / w;
  const uy = ey / w;
  const px = -uy;
  const py = ux;
  const cx = (o.x + i.x) * 0.5;
  const cy = (o.y + i.y) * 0.5;
  const dx = c.x - cx;
  const dy = c.y - cy;
  return { along: (dx * ux + dy * uy) / w, perp: (dx * px + dy * py) / w };
}

function extractFeatures(lm, blends, mat, vw, vh) {
  const L = irisLocal(lm, LEFT, vw, vh);
  const R = irisLocal(lm, RIGHT, vw, vh);
  const b = blendMap(blends);
  const lookX =
    ((b.eyeLookOutLeft || 0) + (b.eyeLookInRight || 0) -
      (b.eyeLookInLeft || 0) - (b.eyeLookOutRight || 0)) *
    0.5;
  const lookY =
    ((b.eyeLookDownLeft || 0) + (b.eyeLookDownRight || 0) -
      (b.eyeLookUpLeft || 0) - (b.eyeLookUpRight || 0)) *
    0.5;
  const pose = euler(mat);
  const a = lm[LEFT.o];
  const bR = lm[RIGHT.o];
  const scale = Math.hypot(a.x - bR.x, a.y - bR.y);
  const hx = (a.x + bR.x) * 0.5;
  const hy = (a.y + bR.y) * 0.5;
  const tz = mat?.data?.[14] ?? 0;
  const irisMean = (L.along + R.along) * 0.5;
  const f = [
    L.along,
    L.perp,
    R.along,
    R.perp,
    lookX,
    lookY,
    pose.yaw,
    pose.pitch,
    pose.roll,
    hx,
    hy,
    scale,
    irisMean,
    tz,
  ];
  if (!f.every(Number.isFinite)) return null;
  return f;
}

function blinkScore(blends, lm, vw, vh) {
  const b = blendMap(blends);
  const blend = ((b.eyeBlinkLeft || 0) + (b.eyeBlinkRight || 0)) * 0.5;
  const e = (ear(lm, LEFT, vw, vh) + ear(lm, RIGHT, vw, vh)) * 0.5;
  const earC = 1 - Math.max(0, Math.min(1, (e - 0.09) / 0.2));
  return 0.7 * blend + 0.3 * earC;
}

function sampleQuality(f, blinkVal, prevLookY) {
  if (!f || f.length !== BASE_FEAT_DIM) return { ok: false, reason: 'features' };
  if (blinkVal > 0.38) return { ok: false, reason: 'blink' };
  if (f[11] < 0.04) return { ok: false, reason: 'face small' };
  if (prevLookY != null && Math.abs(f[5] - prevLookY) > 0.22) return { ok: false, reason: 'unstable' };
  return { ok: true };
}

function videoDeviceId() {
  const tr = state.stream?.getVideoTracks?.()?.[0];
  return tr?.getSettings?.()?.deviceId || tr?.label || '';
}

function geometryFingerprint() {
  const vw = cam.videoWidth || 0;
  const vh = cam.videoHeight || 0;
  const box = stage.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  return {
    v: STORE_VERSION,
    vw,
    vh,
    sw: Math.round(box.width),
    sh: Math.round(box.height),
    dpr,
    deviceId: videoDeviceId(),
    screenX: Math.round(window.screenX ?? 0),
    screenY: Math.round(window.screenY ?? 0),
  };
}

function bumpControlEpoch() {
  state.controlEpoch++;
  state.lastStepAck = -1;
}

function freezeMotor() {
  state.fly.vx = 0;
  state.fly.vy = 0;
}

function motorFresh(now = performance.now()) {
  if (!state.lastMotorCmdMs) return false;
  return now - state.lastMotorCmdMs <= MOTOR_CMD_MAX_MS;
}

function syncUIButtons() {
  const camOn = !!state.stream;
  const inProtocol = state.mode === 'calibrate' || state.mode === 'accuracy';
  $('btnCamStart').disabled = camOn || state.camStarting;
  $('btnCamStop').disabled = !camOn && !state.camStarting;
  $('btnCalib').disabled = !camOn || inProtocol;
  $('btnCancelProto').disabled = !inProtocol;
  $('btnResetCalib').disabled = !camOn && !state.model;
  const geomOk = accuracyProtocolAllowed({
    model: state.model,
    geomStale: state.geomStale,
    storedGeom: state.geom,
    currentGeom: geometryFingerprint(),
  });
  const canAccuracy = geomOk && camOn && state.face && !inProtocol;
  $('btnAccuracy').disabled = !canAccuracy;
  $('btnPause').textContent = state.paused ? 'Resume' : 'Pause';
}

function saveCalibration() {
  try {
    const payload = {
      v: STORE_VERSION,
      geom: state.geom,
      model: state.model,
      heldout: state.heldout,
      headRef: state.headRef,
      savedAt: Date.now(),
    };
    localStorage.setItem(STORE_KEY, JSON.stringify(payload));
  } catch {
    /* quota */
  }
}

function loadCalibration() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);
    if (data.v !== STORE_VERSION || !validateGazeModel(data.model)) return;
    if (!geomMatchesStored(data.geom, geometryFingerprint())) {
      state.geomStale = true;
      return;
    }
    state.model = data.model;
    state.heldout = data.heldout || null;
    state.headRef = data.headRef || null;
    state.geom = data.geom;
    state.geomStale = false;
    state.uncertaintyPx = state.heldout?.medianPx ?? null;
  } catch {
    /* ignore */
  }
}

function forgetCalibration() {
  localStorage.removeItem(STORE_KEY);
  abortProtocols();
  bumpControlEpoch();
  state.model = null;
  state.mode = 'idle';
  state.heldout = null;
  state.dwellBuckets = [];
  state.uncertaintyPx = null;
  state.headRef = null;
  state.geomStale = false;
  state.geom = null;
  clearGazeOutput();
  state.smoothX.reset();
  state.smoothY.reset();
  syncUIButtons();
}

function noteHeadDrift(feats) {
  const ref = state.headRef;
  if (!ref || !feats || state.geomStale) return;
  const scale = feats[11];
  const ds = ref.scale > 1e-5 ? Math.abs(scale - ref.scale) / ref.scale : 0;
  const dy = Math.abs(feats[6] - ref.yaw);
  const dp = Math.abs(feats[7] - ref.pitch);
  if (ds > 0.14 || dy > 0.38 || dp > 0.38) {
    state.geomStale = true;
    setStatus('Head pose drifted — recalibrate for reliable gaze.', 'warn');
  }
}

function stageRect() {
  const pad = 28;
  const w = state.stageCss.w;
  const h = state.stageCss.h;
  return { x: pad, y: pad, w: Math.max(60, w - pad * 2), h: Math.max(60, h - pad * 2) };
}

function normToPx(nx, ny) {
  const r = stageRect();
  return { x: r.x + nx * r.w, y: r.y + ny * r.h };
}

function pxToNorm(x, y) {
  const r = stageRect();
  return { x: (x - r.x) / r.w, y: (y - r.y) / r.h };
}

function resizeStage() {
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const box = stage.getBoundingClientRect();
  const w = Math.max(1, Math.round(box.width));
  const h = Math.max(1, Math.round(box.height));
  if (stage.width !== w * dpr || stage.height !== h * dpr) {
    stage.width = w * dpr;
    stage.height = h * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  state.stageCss = { w, h, dpr };
  const g = geometryFingerprint();
  if (state.geom && !geomMatchesStored(state.geom, g)) state.geomStale = true;
}

function overlayShow(title, body, actions = []) {
  overlay.classList.remove('hidden');
  overlay.innerHTML = `<div class="card"><h2>${title}</h2><p>${body}</p>${actions
    .map((a) => `<button type="button" data-act="${a.id}">${a.label}</button>`)
    .join(' ')}</div>`;
  for (const btn of overlay.querySelectorAll('button[data-act]')) {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-act');
      if (id === 'cancelCalib') cancelCalibration();
      if (id === 'retryDwell') retryDwell();
      if (id === 'cancelAccuracy') cancelAccuracy();
      if (id === 'retryAccuracy') retryAccuracyTarget();
      if (id !== 'cancelCalib' && id !== 'cancelAccuracy') overlayHide();
    });
  }
}

function overlayHide() {
  overlay.classList.add('hidden');
  overlay.innerHTML = '';
}

function abortProtocols() {
  state.calib = null;
  state.accuracy = null;
  if (state.mode === 'calibrate' || state.mode === 'accuracy') {
    state.mode = state.model ? 'track' : 'idle';
    bumpControlEpoch();
  }
  overlayHide();
}

function calibratedGeometryOk() {
  return accuracyProtocolAllowed({
    model: state.model,
    geomStale: state.geomStale,
    storedGeom: state.geom,
    currentGeom: geometryFingerprint(),
  });
}

function cancelAccuracyForGeometry() {
  if (!state.accuracy) return;
  cancelAccuracy();
  setStatus('Display geometry changed — accuracy check cancelled. Previous accuracy report kept.', 'warn');
}

function startCalibration() {
  if (!state.stream) return;
  abortProtocols();
  introEl.classList.add('visually-hidden');
  resizeStage();
  bumpControlEpoch();
  state.mode = 'calibrate';
  state.calib = {
    points: buildCalibSequence(),
    index: 0,
    phase: 'settle',
    tPhase: performance.now(),
    bucket: [],
    dwellBuckets: [],
    fails: 0,
    collecting: false,
    collectStart: 0,
  };
  syncUIButtons();
}

function advanceCalibPoint() {
  const c = state.calib;
  if (!c) return;
  c.index++;
  if (c.index >= c.points.length) {
    finishCalibration();
    return;
  }
  c.phase = 'settle';
  c.tPhase = performance.now();
  c.bucket = [];
  c.collecting = false;
  c.fails = 0;
}

function retryDwell() {
  const c = state.calib;
  if (!c) return;
  c.phase = 'settle';
  c.tPhase = performance.now();
  c.bucket = [];
  c.collecting = false;
  c.fails = 0;
}

function cancelCalibration() {
  const hadModel = !!state.model;
  state.calib = null;
  state.mode = hadModel ? 'track' : 'idle';
  bumpControlEpoch();
  overlayHide();
  setStatus(hadModel ? 'Calibration cancelled. Previous model kept.' : 'Calibration cancelled.');
  syncUIButtons();
}

function finishCalibration() {
  const c = state.calib;
  const samples = [];
  for (const bucket of c.dwellBuckets) {
    for (const s of bucket) samples.push(s);
  }
  const model = fitGazeModel(samples, { dwells: c.dwellBuckets.map((b) => b.map((s) => ({ f: s.f, tx: s.tx, ty: s.ty }))) });
  state.model = model;
  if (model) {
    state.heldout = null;
    state.uncertaintyPx = null;
  }
  state.dwellBuckets = c.dwellBuckets;
  const last = samples[samples.length - 1]?.f;
  if (last) state.headRef = { scale: last[11], yaw: last[6], pitch: last[7] };
  state.calib = null;
  state.mode = model ? 'track' : 'idle';
  resizeStage();
  state.geom = geometryFingerprint();
  state.geomStale = false;
  saveCalibration();
  state.smoothX.reset();
  state.smoothY.reset();
  bumpControlEpoch();
  overlayHide();
  setStatus(
    model
      ? `Calibration saved (${samples.length} samples). Run Check accuracy before trusting gaze.`
      : 'Calibration failed — not enough quality samples. Try again with steady lighting.',
    model ? '' : 'err',
  );
  if (model) introEl.classList.add('visually-hidden');
  syncUIButtons();
}

function calibTargetNorm() {
  const c = state.calib;
  if (!c || c.index < 0 || c.index >= c.points.length) return null;
  return { x: c.points[c.index][0], y: c.points[c.index][1] };
}

function tickCalibration(now, qualityOk, feats) {
  const c = state.calib;
  if (!c || c.index >= c.points.length) return;
  const tgt = calibTargetNorm();
  if (!tgt) return;
  const elapsed = now - c.tPhase;
  if (c.phase === 'settle') {
    if (elapsed < SETTLE_MS) return;
    c.phase = 'collect';
    c.collecting = true;
    c.collectStart = now;
    c.tPhase = now;
    return;
  }
  if (c.phase !== 'collect' || !c.collecting) return;
  const collectElapsed = now - c.collectStart;
  if (qualityOk && feats) {
    c.bucket.push({ f: feats.slice(), tx: tgt.x, ty: tgt.y });
  }
  const longEnough = collectElapsed >= COLLECT_MIN_MS + Math.random() * (COLLECT_MAX_MS - COLLECT_MIN_MS);
  if (longEnough) {
    if (c.bucket.length >= MIN_DWELL_SAMPLES) {
      c.dwellBuckets.push(c.bucket);
      advanceCalibPoint();
    } else {
      c.fails++;
      if (c.fails >= MAX_DWELL_FAILS) {
        overlayShow('Poor samples', 'Could not collect enough stable frames. Adjust lighting or head position.', [
          { id: 'retryDwell', label: 'Retry dot' },
          { id: 'cancelCalib', label: 'Cancel' },
        ]);
        c.phase = 'wait-retry';
      } else {
        c.phase = 'settle';
        c.tPhase = now;
        c.bucket = [];
      }
    }
  }
}

const MAX_ACCURACY_TARGET_FAILS = 4;

function startAccuracy() {
  if (!state.model) return;
  if (!calibratedGeometryOk()) {
    setStatus('Recalibrate for current window and camera before checking accuracy.', 'warn');
    return;
  }
  if (state.calib) cancelCalibration();
  bumpControlEpoch();
  state.mode = 'accuracy';
  state.accuracy = {
    points: HOLDOUT_NORM.slice(),
    index: 0,
    phase: 'settle',
    tPhase: performance.now(),
    currentErrors: [],
    perTarget: [],
    fails: 0,
    collectStart: 0,
  };
  syncUIButtons();
}

function retryAccuracyTarget() {
  const a = state.accuracy;
  if (!a) return;
  a.phase = 'settle';
  a.tPhase = performance.now();
  a.currentErrors = [];
  a.fails = 0;
}

function cancelAccuracy() {
  state.accuracy = null;
  state.mode = state.model ? 'track' : 'idle';
  bumpControlEpoch();
  overlayHide();
  setStatus(
    state.heldout
      ? 'Accuracy check cancelled. Previous accuracy report kept.'
      : 'Accuracy check cancelled. Gaze is calibrated; accuracy not checked.',
  );
  syncUIButtons();
}

function tickAccuracy(now, qualityOk, feats) {
  const a = state.accuracy;
  if (!a) return;
  if (!calibratedGeometryOk()) {
    cancelAccuracyForGeometry();
    return;
  }
  const tgt = a.points[a.index];
  if (!tgt) {
    finishAccuracy(true);
    return;
  }
  const elapsed = now - a.tPhase;
  if (a.phase === 'settle') {
    if (elapsed < SETTLE_MS) return;
    a.phase = 'collect';
    a.collectStart = now;
    a.tPhase = now;
    a.currentErrors = [];
    return;
  }
  if (a.phase === 'wait-retry') return;
  if (a.phase === 'collect') {
    const collectElapsed = now - a.collectStart;
    if (qualityOk && feats && state.model) {
      const raw = predictModel(state.model, feats);
      if (raw && Number.isFinite(raw.x) && Number.isFinite(raw.y)) {
        const px = normToPx(raw.x, raw.y);
        const tpx = normToPx(tgt[0], tgt[1]);
        a.currentErrors.push(Math.hypot(px.x - tpx.x, px.y - tpx.y));
      }
    }
    const longEnough =
      collectElapsed >= COLLECT_MIN_MS + Math.random() * (COLLECT_MAX_MS - COLLECT_MIN_MS);
    if (longEnough) {
      if (a.currentErrors.length >= MIN_DWELL_SAMPLES) {
        a.perTarget.push(a.currentErrors.slice());
        a.index++;
        a.fails = 0;
        a.currentErrors = [];
        if (a.index >= a.points.length) finishAccuracy(true);
        else {
          a.phase = 'settle';
          a.tPhase = now;
        }
      } else {
        a.fails++;
        if (a.fails >= MAX_ACCURACY_TARGET_FAILS) {
          overlayShow(
            'Not enough samples',
            `Target ${a.index + 1} of ${a.points.length}: need at least ${MIN_DWELL_SAMPLES} stable frames.`,
            [{ id: 'retryAccuracy', label: 'Retry target' }, { id: 'cancelAccuracy', label: 'Cancel' }],
          );
          a.phase = 'wait-retry';
        } else {
          a.phase = 'settle';
          a.tPhase = now;
          a.currentErrors = [];
        }
      }
    }
  }
}

function finishAccuracy(allPassed) {
  const a = state.accuracy;
  if (!a) return;
  if (!calibratedGeometryOk()) {
    cancelAccuracyForGeometry();
    return;
  }
  if (!allPassed || a.perTarget.length !== a.points.length) {
    cancelAccuracy();
    return;
  }
  const allErrors = [];
  for (const bucket of a.perTarget) allErrors.push(...bucket);
  if (allErrors.length < a.points.length * MIN_DWELL_SAMPLES) {
    cancelAccuracy();
    return;
  }
  const r = stageRect();
  const diag = Math.hypot(r.w, r.h);
  const sum = summarizeErrors(allErrors, diag);
  state.heldout = { ...sum, at: Date.now(), targets: a.perTarget.length };
  state.uncertaintyPx = sum.medianPx;
  saveCalibration();
  state.accuracy = null;
  state.mode = 'track';
  bumpControlEpoch();
  overlayHide();
  syncUIButtons();
  const msg = `Holdout: median error ${sum.medianPx.toFixed(0)}px (${sum.medianPct.toFixed(1)}% diag), p90 ${sum.p90Px.toFixed(0)}px — ${sum.count} samples across ${a.points.length} targets.`;
  if (sum.medianPct > 12) {
    setStatus(`${msg} Consider recalibrating (lighting / head position).`, 'warn');
  } else {
    setStatus(msg);
  }
}

async function fetchStatus() {
  try {
    const r = await fetch('/api/status', { cache: 'no-store' });
    if (!r.ok) throw new Error(`status ${r.status}`);
    const j = await r.json();
    state.backend.ready = !!j.ready;
    state.backend.training = !!j.training;
    state.backend.progress = j.progress ?? 0;
    state.backend.error = j.error || null;
    state.backend.model = j.model || null;
    return j;
  } catch (e) {
    state.backend.ready = false;
    state.backend.error = e.message;
    return null;
  }
}

async function trainBackend() {
  try {
    const r = await fetch('/api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    if (r.status === 409) {
      setStatus('Training already in progress.', 'warn');
      return;
    }
    if (!r.ok && r.status !== 202) throw new Error(`train ${r.status}`);
    $('btnTrain').disabled = true;
    pollTraining();
  } catch (e) {
    setStatus(`Train failed: ${e.message}`, 'err');
  }
}

function pollTraining() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    const j = await fetchStatus();
    if (!j) return;
    if (j.training) {
      setStatus(`Training fly controller… ${Math.round((j.progress || 0) * 100)}%`);
      return;
    }
    clearInterval(state.pollTimer);
    $('btnTrain').disabled = false;
    if (j.error) setStatus(`Training error: ${j.error}`, 'err');
    else setStatus(j.ready ? 'Fly controller ready.' : 'Training finished; backend not ready.');
  }, 1000);
}

async function postPause() {
  try {
    await fetch('/api/pause', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  } catch {
    /* ignore */
  }
}

async function sendStep(dx, dy, seq) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), STEP_TIMEOUT_MS);
  try {
    const r = await fetch('/api/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dx, dy, seq }),
      signal: ctrl.signal,
    });
    clearTimeout(t);
    if (r.status === 409) return null;
    if (!r.ok) return null;
    const j = await r.json();
    if (j.seq !== seq) return null;
    return j;
  } catch {
    clearTimeout(t);
    return null;
  }
}

function gazeTargetNorm() {
  if (state.pointer && state.pointerNorm) return state.pointerNorm;
  return state.gazeNorm;
}

function trackingInputValid() {
  if (document.hidden) return false;
  if (state.stopFrozen || state.paused || state.mode === 'calibrate' || state.mode === 'accuracy') {
    return false;
  }
  if (!state.backend.ready) return false;
  if (state.pointer) return !!state.pointerNorm;
  if (!state.stream) return false;
  if (!state.model || state.geomStale) return false;
  if (!state.face) return false;
  if (state.blink) return false;
  if (performance.now() - state.lastGoodFeatMs > STALE_MS) return false;
  return !!state.gazeNorm;
}

let prevLookY = null;

function clearGazeOutput() {
  state.gazeNorm = null;
  state.gazePx = null;
}

function updateGazeFromFeatures(feats, tSec) {
  if (!state.model || state.geomStale) {
    clearGazeOutput();
    return;
  }
  const pred = predictModel(state.model, feats);
  if (!pred) {
    clearGazeOutput();
    return;
  }
  const nx = Math.max(0, Math.min(1, state.smoothX.filter(pred.x, tSec)));
  const ny = Math.max(0, Math.min(1, state.smoothY.filter(pred.y, tSec)));
  state.gazeNorm = { x: nx, y: ny };
  state.gazePx = normToPx(nx, ny);
}

async function stepLoop() {
  if (state.stepInFlight) return;
  if (!trackingInputValid()) return;
  const now = performance.now();
  if (now - state.lastStepSent < 1000 / STEP_HZ) return;
  const tgt = gazeTargetNorm();
  if (!tgt) return;
  const { dx, dy } = motorStepDelta(tgt.x, tgt.y, state.fly.x, state.fly.y, MOTOR_INPUT_GAIN);
  const seq = ++state.stepSeq;
  const epoch = state.controlEpoch;
  state.lastStepSent = now;
  state.stepInFlight = true;
  let resp = null;
  try {
    resp = await sendStep(dx, dy, seq);
  } finally {
    state.stepInFlight = false;
  }
  const age = performance.now() - state.lastStepSent;
  if (
    epoch !== state.controlEpoch ||
    !trackingInputValid() ||
    age > STEP_RESPONSE_MAX_MS ||
    !resp ||
    resp.seq !== seq
  ) {
    freezeMotor();
    return;
  }
  state.lastStepAck = seq;
  state.backend.latencyMs = resp.latency_ms;
  const vx = resp.vx;
  const vy = resp.vy;
  state.fly.vx = Number.isFinite(vx) ? vx : 0;
  state.fly.vy = Number.isFinite(vy) ? vy : 0;
  state.lastMotorCmdMs = performance.now();
}

function integrateFly(dt) {
  if (!trackingInputValid() || !motorFresh()) {
    freezeMotor();
    return;
  }
  state.fly.x += state.fly.vx * FLY_SPEED * dt;
  state.fly.y += state.fly.vy * FLY_SPEED * dt;
  state.fly.x = Math.max(0.02, Math.min(0.98, state.fly.x));
  state.fly.y = Math.max(0.02, Math.min(0.98, state.fly.y));
}

function draw(now) {
  resizeStage();
  const w = state.stageCss.w;
  const h = state.stageCss.h;
  ctx.clearRect(0, 0, w, h);

  const r = stageRect();
  ctx.strokeStyle = '#2a3140';
  ctx.lineWidth = 1;
  ctx.strokeRect(r.x, r.y, r.w, r.h);

  if (state.mode === 'calibrate' && state.calib) {
    const c = state.calib;
    const tgt = calibTargetNorm();
    if (tgt) {
      const p = normToPx(tgt.x, tgt.y);
      ctx.fillStyle = '#5ee4c7';
      ctx.beginPath();
      ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#e8eaef';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 20, 0, Math.PI * 2);
      ctx.stroke();
      const phaseLabel =
        c.phase === 'settle'
          ? 'Settle'
          : c.phase === 'wait-retry'
            ? 'Retry'
            : 'Collecting';
      ctx.font = '600 14px system-ui';
      ctx.fillStyle = '#e8eaef';
      ctx.textAlign = 'left';
      ctx.fillText(
        `Calibration ${c.index + 1} / ${c.points.length} — ${phaseLabel}`,
        r.x,
        r.y - 22,
      );
      ctx.font = '14px system-ui';
      ctx.fillStyle = '#8b909c';
      ctx.fillText('Look at the dot (keep head comfortable)', r.x, r.y - 6);
    }
  }

  if (state.mode === 'accuracy' && state.accuracy) {
    const a = state.accuracy;
    const pt = a.points[a.index];
    if (pt) {
      const p = normToPx(pt[0], pt[1]);
      ctx.fillStyle = '#f5b942';
      ctx.beginPath();
      ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
      ctx.fill();
      const phaseLabel =
        a.phase === 'settle' ? 'Settle' : a.phase === 'wait-retry' ? 'Retry' : 'Collecting';
      ctx.font = '600 14px system-ui';
      ctx.fillStyle = '#e8eaef';
      ctx.textAlign = 'left';
      ctx.fillText(`Accuracy ${a.index + 1} / ${a.points.length} — ${phaseLabel}`, r.x, r.y - 22);
      ctx.font = '14px system-ui';
      ctx.fillStyle = '#8b909c';
      ctx.fillText('Look at the dot', r.x, r.y - 6);
    }
  }

  if (state.gazePx && !state.pointer) {
    const g = state.gazePx;
    ctx.strokeStyle = 'rgba(94, 228, 199, 0.85)';
    ctx.lineWidth = 2;
    if (state.heldout?.medianPx != null && state.uncertaintyPx != null) {
      ctx.beginPath();
      ctx.arc(g.x, g.y, Math.max(6, state.uncertaintyPx), 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.beginPath();
    ctx.arc(g.x, g.y, 4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(94, 228, 199, 0.35)';
    ctx.fill();
    ctx.stroke();
  }

  if (state.pointer && state.pointerNorm) {
    const p = normToPx(state.pointerNorm.x, state.pointerNorm.y);
    ctx.strokeStyle = 'rgba(94, 228, 199, 0.5)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  const flyPx = normToPx(state.fly.x, state.fly.y);
  ctx.font = '28px system-ui';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.shadowColor = 'rgba(245,185,66,0.45)';
  ctx.shadowBlur = 12;
  ctx.fillText('🪰', flyPx.x, flyPx.y);
  ctx.shadowBlur = 0;

  if (state.pointer) {
    ctx.font = '12px system-ui';
    ctx.fillStyle = '#f5b942';
    ctx.textAlign = 'left';
    ctx.fillText('Pointer test — backend drive', r.x, r.y + r.h + 18);
  }

  const m = state.backend.model;
  const neu = m ? `${m.neurons ?? '—'} neurons` : 'backend offline';
  const lat = state.backend.latencyMs != null ? `${state.backend.latencyMs.toFixed(0)}ms` : '—';
  let gazeTxt = 'uncalibrated';
  if (state.model && !state.geomStale) {
    gazeTxt = state.heldout
      ? `median error ~${(state.uncertaintyPx ?? state.heldout.medianPx).toFixed(0)}px`
      : 'accuracy not checked';
  }
  telemetryEl.textContent = `Backend: ${neu} · ready ${state.backend.ready ? 'yes' : 'no'} · latency ${lat} · Gaze: ${gazeTxt} · infer ${state.stats.inferFps.toFixed(0)}fps · ${state.stats.delegate}`;

  drawPip();
}

let drawN = 0;
let drawT0 = performance.now();
function renderLoop(now) {
  drawN++;
  if (now - drawT0 > 500) {
    state.stats.drawFps = (drawN * 1000) / (now - drawT0);
    drawN = 0;
    drawT0 = now;
  }
  let dt = 0;
  if (state.lastRafTs > 0) {
    dt = Math.min(RAF_DT_MAX_S, Math.max(0, (now - state.lastRafTs) / 1000));
  }
  state.lastRafTs = now;
  if (trackingInputValid() && motorFresh(now)) integrateFly(dt);
  else freezeMotor();
  draw(now);
  stepLoop();
  state.rafId = requestAnimationFrame(renderLoop);
}

function drawPip() {
  if (!state.stream || state.mode === 'calibrate' || state.mode === 'accuracy' || $('hidePip').checked) {
    pip.classList.add('hidden');
    return;
  }
  pip.classList.remove('hidden');
  pctx.fillStyle = '#000';
  pctx.fillRect(0, 0, pip.width, pip.height);
  if (cam.readyState >= 2 && cam.videoWidth > 0) {
    const scale = Math.max(pip.width / cam.videoWidth, pip.height / cam.videoHeight);
    const dw = cam.videoWidth * scale;
    const dh = cam.videoHeight * scale;
    const dx = (pip.width - dw) * 0.5;
    const dy = (pip.height - dh) * 0.5;
    pctx.drawImage(cam, dx, dy, dw, dh);
  }
}

let inferN = 0;
let inferT0 = performance.now();

function onVideoFrame(now, metadata, gen) {
  if (!state.inferScheduled || gen !== state.cameraGen) return;
  state.inferScheduled = false;
  scheduleInfer(gen);
  if (!state.landmarker || cam.readyState < 2) return;
  const ts = metadata?.mediaTime != null ? metadata.mediaTime * 1000 : (cam.currentTime || 0) * 1000;
  if (ts <= state.lastVideoTs) return;
  state.lastVideoTs = ts;
  const t0 = performance.now();
  let result;
  try {
    result = state.landmarker.detectForVideo(cam, ts);
  } catch {
    return;
  }
  state.stats.inferMs = state.stats.inferMs * 0.85 + (performance.now() - t0) * 0.15;
  inferN++;
  const tnow = performance.now();
  if (tnow - inferT0 > 500) {
    state.stats.inferFps = (inferN * 1000) / (tnow - inferT0);
    inferN = 0;
    inferT0 = tnow;
  }

  const lm = result.faceLandmarks?.[0];
  const blends = result.faceBlendshapes?.[0];
  const mat = result.facialTransformationMatrixes?.[0];
  const tSec = tnow / 1000;

  const vw = cam.videoWidth || 1;
  const vh = cam.videoHeight || 1;

  if (!lm) {
    if (state.prevFace) bumpControlEpoch();
    state.prevFace = false;
    state.face = false;
    state.blink = false;
    state.prevBlink = false;
    state.feats = null;
    clearGazeOutput();
    state.smoothX.reset();
    state.smoothY.reset();
    if (state.mode === 'calibrate') tickCalibration(tnow, false, null);
    if (state.mode === 'accuracy') tickAccuracy(tnow, false, null);
    syncUIButtons();
    return;
  }
  state.face = true;
  if (!state.prevFace) syncUIButtons();
  state.prevFace = true;
  const blinkVal = blinkScore(blends, lm, vw, vh);
  const blinkNow = blinkVal > 0.38;
  if (blinkNow && !state.prevBlink) bumpControlEpoch();
  state.prevBlink = blinkNow;
  state.blink = blinkNow;
  const feats = extractFeatures(lm, blends, mat, vw, vh);
  const q = sampleQuality(feats, blinkVal, prevLookY);
  prevLookY = feats?.[5] ?? null;
  if (!q.ok) {
    if (q.reason === 'blink') {
      state.smoothX.reset();
      state.smoothY.reset();
      clearGazeOutput();
    }
    if (state.mode === 'calibrate') tickCalibration(tnow, false, null);
    if (state.mode === 'accuracy') tickAccuracy(tnow, false, null);
    return;
  }
  prevLookY = feats[5];
  state.feats = feats;
  state.lastGoodFeatMs = tnow;
  noteHeadDrift(feats);

  if (state.mode === 'calibrate') {
    tickCalibration(tnow, true, feats);
    return;
  }
  if (state.mode === 'accuracy') {
    tickAccuracy(tnow, true, feats);
    return;
  }
  if (!state.pointer) updateGazeFromFeatures(feats, tSec);
}

function scheduleInfer(gen = state.cameraGen) {
  if (!state.loopsOn || gen !== state.cameraGen || document.hidden) return;
  state.inferScheduled = true;
  if (cam.requestVideoFrameCallback) {
    cam.requestVideoFrameCallback((now, meta) => onVideoFrame(now, meta, gen));
  } else requestAnimationFrame(() => onVideoFrame(performance.now(), {}, gen));
}

function stopMediaStream(stream) {
  if (!stream) return;
  for (const tr of stream.getTracks()) {
    tr.onended = null;
    tr.stop();
  }
}

function closeLandmarkerInstance(lm) {
  if (!lm) return;
  try {
    lm.close();
  } catch {
    /* ignore */
  }
}

function releaseCameraTracks() {
  closeLandmarkerInstance(state.landmarker);
  state.landmarker = null;
  stopMediaStream(state.stream);
  state.stream = null;
  cam.srcObject = null;
}

function abandonStaleStartup(sessionGen, stream, landmarker) {
  if (cameraSessionOwned(sessionGen, state.cameraGen)) return false;
  stopMediaStream(stream);
  closeLandmarkerInstance(landmarker);
  if (state.stream === stream) {
    state.stream = null;
    cam.srcObject = null;
  }
  if (state.landmarker === landmarker) state.landmarker = null;
  return true;
}

function resetVisionState() {
  state.lastVideoTs = 0;
  state.inferScheduled = false;
  state.face = false;
  state.prevFace = false;
  state.prevBlink = false;
  state.blink = false;
  state.feats = null;
  prevLookY = null;
  clearGazeOutput();
  state.smoothX.reset();
  state.smoothY.reset();
  freezeMotor();
  state.lastMotorCmdMs = 0;
}

function onTrackEnded() {
  if (!state.stream) return;
  setStatus('Camera track ended — restart camera when ready.', 'warn');
  stopCamera();
}

async function startCamera() {
  if (state.stream || state.camStarting) return;
  state.camStarting = true;
  state.cameraGen++;
  const gen = state.cameraGen;
  syncUIButtons();
  setStatus('Requesting camera…');
  let stream = null;
  let landmarker = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: 'user',
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30, min: 15 },
      },
    });
    if (abandonStaleStartup(gen, stream, null)) return;
    resetVisionState();
    abortProtocols();
    state.stream = stream;
    const vtr = stream.getVideoTracks()[0];
    if (vtr) vtr.onended = onTrackEnded;
    cam.srcObject = stream;
    await cam.play();
    if (abandonStaleStartup(gen, stream, null)) return;
    const origin = location.origin;
    const fileset = await FilesetResolver.forVisionTasks(`${origin}/vendor/wasm`);
    if (abandonStaleStartup(gen, stream, null)) return;
    const base = {
      baseOptions: {
        modelAssetPath: `${origin}/vendor/face_landmarker.task`,
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    };
    try {
      landmarker = await FaceLandmarker.createFromOptions(fileset, base);
      state.stats.delegate = 'GPU';
    } catch {
      if (abandonStaleStartup(gen, stream, null)) return;
      base.baseOptions.delegate = 'CPU';
      landmarker = await FaceLandmarker.createFromOptions(fileset, base);
      state.stats.delegate = 'CPU';
    }
    if (abandonStaleStartup(gen, stream, landmarker)) {
      landmarker = null;
      return;
    }
    state.landmarker = landmarker;
    landmarker = null;
    state.stopFrozen = false;
    state.geom = geometryFingerprint();
    loadCalibration();
    if (state.geomStale) {
      setStatus(
        'Place this window on the screen you will use, then recalibrate for this camera and position.',
        'warn',
      );
    } else if (!state.model) {
      setStatus('Camera on. Keep the window on-screen, then run Calibrate (~40s).', '');
    } else if (!state.heldout) {
      setStatus('Calibration restored. Run Check accuracy before trusting gaze.', 'warn');
    } else {
      setStatus('Camera on. Gaze model restored from this device.');
      introEl.classList.add('visually-hidden');
    }
    state.loopsOn = true;
    scheduleInfer(gen);
    bumpControlEpoch();
    syncUIButtons();
  } catch (e) {
    if (cameraSessionOwned(gen, state.cameraGen)) {
      releaseCameraTracks();
      resetVisionState();
      setStatus(
        e.name === 'NotAllowedError' ? 'Camera permission denied — allow camera and retry.' : e.message,
        'err',
      );
    } else {
      abandonStaleStartup(gen, stream, landmarker);
    }
  } finally {
    if (!cameraSessionOwned(gen, state.cameraGen)) {
      abandonStaleStartup(gen, stream, landmarker);
    } else if (landmarker) {
      closeLandmarkerInstance(landmarker);
    }
    if (cameraSessionOwned(gen, state.cameraGen)) state.camStarting = false;
    syncUIButtons();
  }
}

function stopCamera() {
  state.cameraGen++;
  state.camStarting = false;
  state.loopsOn = false;
  state.inferScheduled = false;
  abortProtocols();
  releaseCameraTracks();
  resetVisionState();
  state.stopFrozen = true;
  bumpControlEpoch();
  state.lastRafTs = 0;
  setStatus('Camera stopped — fly frozen. Resume or switch Pointer test to drive again.');
  syncUIButtons();
}

$('btnCamStart').addEventListener('click', startCamera);
$('btnCamStop').addEventListener('click', stopCamera);
$('btnCalib').addEventListener('click', startCalibration);
$('btnAccuracy').addEventListener('click', startAccuracy);
$('btnTrain').addEventListener('click', trainBackend);
$('btnResetCalib').addEventListener('click', () => {
  forgetCalibration();
  setStatus('Calibration cleared. Run Calibrate again.');
  introEl.classList.remove('visually-hidden');
});
$('btnCancelProto').addEventListener('click', () => {
  if (state.mode === 'calibrate') cancelCalibration();
  else if (state.mode === 'accuracy') cancelAccuracy();
});
$('btnPause').addEventListener('click', async () => {
  state.paused = !state.paused;
  bumpControlEpoch();
  state.lastRafTs = 0;
  freezeMotor();
  if (state.paused) await postPause();
  else state.stopFrozen = false;
  syncUIButtons();
  setStatus(state.paused ? 'Paused — fly frozen.' : 'Resumed.');
});
$('pointerMode').addEventListener('change', (e) => {
  state.pointer = e.target.checked;
  state.pointerNorm = null;
  state.stopFrozen = false;
  bumpControlEpoch();
  state.smoothX.reset();
  state.smoothY.reset();
  clearGazeOutput();
  syncUIButtons();
  if (state.pointer) setStatus('Pointer test on — move mouse over field; uses backend.', 'warn');
});
$('hidePip').addEventListener('change', drawPip);

stage.addEventListener('pointermove', (e) => {
  if (!state.pointer) return;
  const rect = stage.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const n = pxToNorm(x, y);
  state.pointerNorm = { x: Math.max(0, Math.min(1, n.x)), y: Math.max(0, Math.min(1, n.y)) };
  state.gazePx = normToPx(state.pointerNorm.x, state.pointerNorm.y);
});

window.addEventListener('resize', () => {
  resizeStage();
  const g = geometryFingerprint();
  if (state.geom && !geomMatchesStored(state.geom, g)) {
    state.geomStale = true;
    setStatus('Display or window position changed — recalibrate when ready.', 'warn');
  }
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    state.inferScheduled = false;
    bumpControlEpoch();
    freezeMotor();
  } else if (state.stream && state.loopsOn) {
    scheduleInfer(state.cameraGen);
  }
});

window.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  if (state.mode === 'calibrate') cancelCalibration();
  else if (state.mode === 'accuracy') cancelAccuracy();
});

resizeStage();
syncUIButtons();
requestAnimationFrame(renderLoop);
setInterval(fetchStatus, 3000);
fetchStatus();
