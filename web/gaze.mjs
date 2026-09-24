/**
 * Pure numerical gaze mapping: standardized ridge regression with optional
 * low-order gaze nonlinearities. No browser APIs.
 */

export const BASE_FEAT_DIM = 14;
export const STORE_VERSION = 3;

const STD_FLOOR = 1e-3;
const MIN_SAMPLES_LINEAR = 24;
const MIN_SAMPLES_EXTENDED = 80;
const DEFAULT_LAMBDAS = [1e-2, 1e-1, 1, 10, 100];

/** @typedef {{ f: number[], tx: number, ty: number }} LabeledSample */
/** @typedef {{ kind: 'linear'|'extended', mean: number[], std: number[], wx: number[], wy: number[], lambda: number, featDim: number }} GazeModel */

export function isFiniteVector(v) {
  if (!Array.isArray(v) || v.length === 0) return false;
  for (const x of v) if (!Number.isFinite(x)) return false;
  return true;
}

export function assertSample(s) {
  if (!s || !isFiniteVector(s.f)) throw new TypeError('sample features must be finite');
  if (!Number.isFinite(s.tx) || !Number.isFinite(s.ty)) throw new TypeError('sample targets must be finite');
  if (s.f.length !== BASE_FEAT_DIM) throw new RangeError(`expected ${BASE_FEAT_DIM} features`);
}

/**
 * Expand base features for design matrix. Bias is added separately (unpenalized).
 */
export function expandFeatures(base, kind) {
  if (!isFiniteVector(base) || base.length !== BASE_FEAT_DIM) return null;
  if (kind === 'linear') return base.slice();
  if (kind === 'extended') {
    const [lA, lP, rA, rP, lookX, lookY] = base;
    const extra = [
      lA * lA,
      rA * rA,
      lookX * lookX,
      lookY * lookY,
      lA * lookX,
      rA * lookY,
      lP * rP,
    ];
    return base.concat(extra);
  }
  return null;
}

export function designDim(kind) {
  return kind === 'extended' ? BASE_FEAT_DIM + 7 : BASE_FEAT_DIM;
}

function computeStandardizer(samples, kind) {
  const d = designDim(kind);
  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  const n = samples.length;
  if (n < 2) return null;
  const rows = [];
  for (const s of samples) {
    const row = expandFeatures(s.f, kind);
    if (!row) return null;
    rows.push(row);
  }
  for (let j = 0; j < d; j++) {
    let m = 0;
    for (const row of rows) m += row[j];
    m /= n;
    mean[j] = m;
  }
  for (let j = 0; j < d; j++) {
    let v = 0;
    for (const row of rows) {
      const t = row[j] - mean[j];
      v += t * t;
    }
    std[j] = Math.max(STD_FLOOR, Math.sqrt(v / Math.max(1, n - 1)));
  }
  return { mean, std };
}

export function standardize(row, mean, std) {
  const out = new Array(row.length);
  for (let i = 0; i < row.length; i++) {
    out[i] = (row[i] - mean[i]) / std[i];
  }
  return out;
}

/** Stable pivoted Gaussian elimination for square A */
export function solveSquare(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.slice().concat(b[i]));
  for (let col = 0; col < n; col++) {
    let piv = col;
    let best = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      const v = Math.abs(M[r][col]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (best < 1e-12) return null;
    if (piv !== col) [M[col], M[piv]] = [M[piv], M[col]];
    const div = M[col][col];
    for (let j = col; j <= n; j++) M[col][j] /= div;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let j = col; j <= n; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row[n]);
}

function buildNormalEquations(samples, kind, lambda) {
  const d = designDim(kind);
  const stat = computeStandardizer(samples, kind);
  if (!stat) return null;
  const p = d + 1;
  const xtx = Array.from({ length: p }, () => new Array(p).fill(0));
  const yx = new Array(p).fill(0);
  const yy = new Array(p).fill(0);
  for (const s of samples) {
    const raw = expandFeatures(s.f, kind);
    if (!raw) return null;
    const z = standardize(raw, stat.mean, stat.std);
    const row = [1, ...z];
    for (let i = 0; i < p; i++) {
      yx[i] += row[i] * s.tx;
      yy[i] += row[i] * s.ty;
      for (let j = 0; j < p; j++) xtx[i][j] += row[i] * row[j];
    }
  }
  for (let i = 1; i < p; i++) xtx[i][i] += lambda;
  const wx = solveSquare(xtx.map((r) => r.slice()), yx);
  const wy = solveSquare(xtx.map((r) => r.slice()), yy);
  if (!wx || !wy) return null;
  return { stat, wx, wy, kind };
}

export function validateGazeModel(model) {
  if (!model || model.featDim !== BASE_FEAT_DIM) return false;
  if (model.kind !== 'linear' && model.kind !== 'extended') return false;
  const d = designDim(model.kind);
  if (!isFiniteVector(model.mean) || model.mean.length !== d) return false;
  if (!isFiniteVector(model.std) || model.std.length !== d) return false;
  const p = d + 1;
  if (!isFiniteVector(model.wx) || model.wx.length !== p) return false;
  if (!isFiniteVector(model.wy) || model.wy.length !== p) return false;
  if (!Number.isFinite(model.lambda)) return false;
  return true;
}

export function predictModel(model, baseFeats) {
  if (!validateGazeModel(model)) return null;
  const raw = expandFeatures(baseFeats, model.kind);
  if (!raw) return null;
  const z = standardize(raw, model.mean, model.std);
  let x = model.wx[0];
  let y = model.wy[0];
  for (let i = 0; i < z.length; i++) {
    x += model.wx[i + 1] * z[i];
    y += model.wy[i + 1] * z[i];
  }
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

/** Closed-loop sensory gain on normalized position error (see About). */
export const MOTOR_INPUT_GAIN = 4;

export function motorStepDelta(targetX, targetY, flyX, flyY, gain = MOTOR_INPUT_GAIN) {
  if (
    !Number.isFinite(targetX) ||
    !Number.isFinite(targetY) ||
    !Number.isFinite(flyX) ||
    !Number.isFinite(flyY)
  ) {
    return { dx: 0, dy: 0 };
  }
  const dx = Math.max(-1, Math.min(1, gain * (targetX - flyX)));
  const dy = Math.max(-1, Math.min(1, gain * (targetY - flyY)));
  return { dx, dy };
}

/**
 * Choose ridge λ using held-out calibration dwells (leave-one-dwell-out).
 * Never uses external validation targets.
 */
export function chooseLambdaFromDwells(dwells, kind, lambdas = DEFAULT_LAMBDAS) {
  if (!dwells?.length) return lambdas[2];
  let best = lambdas[0];
  let bestScore = Infinity;
  for (const lam of lambdas) {
    let total = 0;
    let parts = 0;
    for (let i = 0; i < dwells.length; i++) {
      const hold = dwells[i];
      const train = [];
      for (let j = 0; j < dwells.length; j++) {
        if (j === i) continue;
        train.push(...dwells[j]);
      }
      if (train.length < MIN_SAMPLES_LINEAR) continue;
      const evalScore = (() => {
        const built = buildNormalEquations(train, kind, lam);
        if (!built) return Infinity;
        const model = {
          kind: built.kind,
          mean: built.stat.mean,
          std: built.stat.std,
          wx: built.wx,
          wy: built.wy,
          lambda: lam,
          featDim: BASE_FEAT_DIM,
        };
        let e = 0;
        for (const s of hold) {
          const p = predictModel(model, s.f);
          if (!p) return Infinity;
          e += (p.x - s.tx) ** 2 + (p.y - s.ty) ** 2;
        }
        return e / hold.length;
      })();
      total += evalScore;
      parts++;
    }
    if (!parts) continue;
    const avg = total / parts;
    if (avg < bestScore) {
      bestScore = avg;
      best = lam;
    }
  }
  return best;
}

export function pickModelKind(sampleCount) {
  return sampleCount >= MIN_SAMPLES_EXTENDED ? 'extended' : 'linear';
}

/**
 * Fit gaze ridge model from labeled calibration samples.
 * @param {LabeledSample[]} samples
 * @param {{ dwells?: LabeledSample[][] }} options
 */
export function fitGazeModel(samples, options = {}) {
  if (!Array.isArray(samples)) throw new TypeError('samples must be an array');
  const clean = [];
  for (const s of samples) {
    try {
      assertSample(s);
      clean.push(s);
    } catch {
      /* skip malformed */
    }
  }
  if (clean.length < MIN_SAMPLES_LINEAR) return null;
  const kind = pickModelKind(clean.length);
  const dwells = options.dwells?.filter((d) => d?.length >= MIN_SAMPLES_LINEAR / 2) ?? [];
  const lambda =
    dwells.length >= 2
      ? chooseLambdaFromDwells(dwells, kind)
      : DEFAULT_LAMBDAS[2];
  const built = buildNormalEquations(clean, kind, lambda);
  if (!built) return null;
  return {
    kind: built.kind,
    mean: built.stat.mean,
    std: built.stat.std,
    wx: built.wx,
    wy: built.wy,
    lambda,
    featDim: BASE_FEAT_DIM,
    sampleCount: clean.length,
  };
}

export function percentile(sortedCopy, p) {
  if (!sortedCopy.length) return 0;
  const s = sortedCopy.slice().sort((a, b) => a - b);
  const idx = Math.max(0, Math.min(s.length - 1, Math.round((s.length - 1) * p)));
  return s[idx];
}

export function summarizeErrors(errorsPx, stageDiag) {
  const med = percentile(errorsPx, 0.5);
  const p90 = percentile(errorsPx, 0.9);
  const pctMed = stageDiag > 0 ? (med / stageDiag) * 100 : 0;
  const pct90 = stageDiag > 0 ? (p90 / stageDiag) * 100 : 0;
  return { medianPx: med, p90Px: p90, medianPct: pctMed, p90Pct: pct90, count: errorsPx.length };
}

const GEOM_POS_TOL_CSS = 40;

/** @param {Record<string, unknown>} a @param {Record<string, unknown>} b */
export function geomMatchesStored(a, b) {
  if (!a || !b || a.v !== b.v) return false;
  const dw = Math.abs((a.vw ?? 0) - (b.vw ?? 0)) > 8 || Math.abs((a.vh ?? 0) - (b.vh ?? 0)) > 8;
  const ds =
    Math.abs((a.sw ?? 0) - (b.sw ?? 0)) > 24 || Math.abs((a.sh ?? 0) - (b.sh ?? 0)) > 24;
  const dpr = a.dpr !== b.dpr;
  const device = (a.deviceId ?? '') !== (b.deviceId ?? '');
  const pos =
    Math.abs((a.screenX ?? 0) - (b.screenX ?? 0)) > GEOM_POS_TOL_CSS ||
    Math.abs((a.screenY ?? 0) - (b.screenY ?? 0)) > GEOM_POS_TOL_CSS;
  return !dw && !ds && !dpr && !device && !pos;
}

/** For tests: generate synthetic dataset with known mapping */
export function syntheticDataset(mapping, n = 200, noise = 0.02) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const f = Array.from({ length: BASE_FEAT_DIM }, () => Math.random() * 2 - 1);
    const t = mapping(f);
    out.push({
      f,
      tx: t.x + (Math.random() - 0.5) * noise,
      ty: t.y + (Math.random() - 0.5) * noise,
    });
  }
  return out;
}
