#!/usr/bin/env python3
"""
Connectome-reservoir gaze motor controller.

Browser-side gaze estimates a normalized position error (dx, dy). This module
drives four real visual-projection cell groups as an artificial directional
stimulus into the fixed MaleCNS LIF network, then reads a linear decoder over
descending-neuron (DN) membrane voltages to produce motor velocities (vx, vy).

This is engineered reservoir readout training — not biological learning of
human gaze, and not a claim that these channel→direction assignments are
natural fly visual directions.

Trial-reset reservoir windows
-----------------------------
Each control (and training) sample is an independent trial:
  1. reset() clears membrane / synapse / delay-line transient state
  2. rng is re-seeded to RESPONSE_RNG_SEED so Poisson drive draws are repeatable
  3. integrate SAMPLE_MS at dt=1 ms
  4. average DN voltage over the final FEATURE_MS of that window

The connectome weights stay frozen; only the linear readout W is fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
UPSTREAM_SCRIPTS = PROJECT_ROOT / "upstream" / "flybrain" / "scripts"
UPSTREAM_COMMIT = "9aaed09db9aef1c961a5aff82e47982e04b5e516"
DEFAULT_CONNECTOME = PROJECT_ROOT / "data" / "processed" / "male-cns-v1.0-traced-typed-min5.npz"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "processed" / "gaze-controller.npz"

CHECKPOINT_VERSION = 2
ENCODER_VERSION = "gaze-drive-v1"
FEATURE_VERSION = "dn-v-mean-final40ms-v1"

# Arbitrary engineered channel map — NOT biological direction claims.
# drive_hz order: [neg_x, pos_x, neg_y, pos_y] * DRIVE_HZ_SCALE
INPUT_GROUPS: tuple[tuple[str, str], ...] = (
    ("LC4", "L"),    # negative x
    ("LC4", "R"),    # positive x
    ("LPLC2", "L"),  # negative y
    ("LPLC2", "R"),  # positive y
)
INPUT_CHANNEL_MEANING = (
    "neg_x->LC4-L",
    "pos_x->LC4-R",
    "neg_y->LPLC2-L",
    "pos_y->LPLC2-R",
)
DRIVE_HZ_SCALE = 140.0
SAMPLE_MS = 80
FEATURE_MS = 40  # final-window average
RESPONSE_RNG_SEED = 11
TRAIN_RNG_SEED = 912
N_TRAIN_RANDOM = 140
N_TRAIN_BASELINE = 4
N_HOLDOUT = 36
RIDGE = 1.0
DEADBAND = 0.015
SCALE_FLOOR = 0.1
ACTIVE_FEATURE_STD = 0.001
ACCEPT_MAE_MAX = 0.20
ACCEPT_SIGN_MIN = 0.85
BALANCE_HEMISPHERES = False

if str(UPSTREAM_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_SCRIPTS))

from flysim import FlyBrain, Params  # noqa: E402

GAZE_PARAMS = Params(
    noise=0.0,
    sign_override=(),
    apl_scale=1.0,
    kc_normalise=False,
    kc_thresh_scale=1.0,
    mbon_hold_frac=0.0,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(obj: Any) -> Any:
    """Normalize nested structures for JSON round-trip equality."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def gaze_params_dict() -> dict[str, Any]:
    return _jsonable(asdict(GAZE_PARAMS))


def _validate_error(dx: float, dy: float) -> tuple[float, float]:
    if not (np.isfinite(dx) and np.isfinite(dy)):
        raise ValueError("dx/dy must be finite")
    dx_f, dy_f = float(dx), float(dy)
    if abs(dx_f) > 1.0 or abs(dy_f) > 1.0:
        raise ValueError("dx/dy must be within [-1, 1]")
    return dx_f, dy_f


def _deadband_clip(vx: float, vy: float) -> tuple[float, float]:
    if abs(vx) < DEADBAND:
        vx = 0.0
    if abs(vy) < DEADBAND:
        vy = 0.0
    return float(np.clip(vx, -1.0, 1.0)), float(np.clip(vy, -1.0, 1.0))


def _require_positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    iv = int(value)
    if iv <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return iv


def _require_nonneg_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    iv = int(value)
    if iv < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return iv


def validate_controller_args(
    *,
    sample_ms: Any,
    feature_ms: Any,
    n_train_random: Any,
    n_train_baseline: Any,
    n_holdout: Any,
    ridge: Any,
    drive_hz_scale: Any,
) -> tuple[int, int, int, int, int, float, float]:
    sample = _require_positive_int("sample_ms", sample_ms)
    feature = _require_positive_int("feature_ms", feature_ms)
    if feature > sample:
        raise ValueError("feature_ms must be <= sample_ms")
    n_rand = _require_nonneg_int("n_train_random", n_train_random)
    n_base = _require_nonneg_int("n_train_baseline", n_train_baseline)
    if n_rand + n_base <= 0:
        raise ValueError("n_train (random+baseline) must be > 0")
    n_hold = _require_positive_int("n_holdout", n_holdout)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float, np.floating, np.integer)):
        raise ValueError("ridge must be a positive finite number")
    ridge_f = float(ridge)
    if not np.isfinite(ridge_f) or ridge_f <= 0.0:
        raise ValueError("ridge must be a positive finite number")
    if isinstance(drive_hz_scale, bool) or not isinstance(
        drive_hz_scale, (int, float, np.floating, np.integer)
    ):
        raise ValueError("drive_hz_scale must be a finite number")
    drive_f = float(drive_hz_scale)
    if not np.isfinite(drive_f):
        raise ValueError("drive_hz_scale must be finite")
    return sample, feature, n_rand, n_base, n_hold, ridge_f, drive_f


class GazeController:
    """
    Fixed-connectome reservoir + linear DN readout for gaze error → velocity.

    Each predict/train sample uses a trial-reset reservoir window (see module
    docstring): reset transients, seed RNG, integrate SAMPLE_MS, average DN
    voltage over the final FEATURE_MS. Decoder sees only those DN features —
    never raw target error.
    """

    def __init__(
        self,
        connectome_path: Path | str | None = None,
        checkpoint_path: Path | str | None = None,
        *,
        n_train_random: int = N_TRAIN_RANDOM,
        n_train_baseline: int = N_TRAIN_BASELINE,
        n_holdout: int = N_HOLDOUT,
        ridge: float = RIDGE,
        sample_ms: int = SAMPLE_MS,
        feature_ms: int = FEATURE_MS,
        drive_hz_scale: float = DRIVE_HZ_SCALE,
        enforce_acceptance: bool = True,
        auto_load_brain: bool = True,
    ) -> None:
        (
            sample_ms,
            feature_ms,
            n_train_random,
            n_train_baseline,
            n_holdout,
            ridge,
            drive_hz_scale,
        ) = validate_controller_args(
            sample_ms=sample_ms,
            feature_ms=feature_ms,
            n_train_random=n_train_random,
            n_train_baseline=n_train_baseline,
            n_holdout=n_holdout,
            ridge=ridge,
            drive_hz_scale=drive_hz_scale,
        )
        self.connectome_path = Path(connectome_path or DEFAULT_CONNECTOME)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT
        self.n_train_random = n_train_random
        self.n_train_baseline = n_train_baseline
        self.n_holdout = n_holdout
        self.ridge = ridge
        self.sample_ms = sample_ms
        self.feature_ms = feature_ms
        self.drive_hz_scale = drive_hz_scale
        self.enforce_acceptance = bool(enforce_acceptance)

        self._lock = threading.RLock()
        self.brain: FlyBrain | None = None
        self._input_groups: list[np.ndarray] = []
        self._input_ids: list[list[int]] = []
        self._dn_idx: np.ndarray = np.zeros(0, dtype=np.int64)
        self._dn_ids: np.ndarray = np.zeros(0, dtype=np.int64)

        self.scales: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.metrics: dict[str, Any] | None = None
        self.model_info: dict[str, Any] | None = None
        self.graph_sha256: str | None = None
        self.ready = False
        self.load_error: str | None = None

        if auto_load_brain:
            self._ensure_brain()
        if self.checkpoint_path.is_file():
            status = self.load_checkpoint(self.checkpoint_path)
            if status != "ready":
                self.load_error = status
                self.ready = False
        else:
            self.ready = False
            self.load_error = "checkpoint absent"

    def inference_settings(self) -> dict[str, Any]:
        """Settings that define the input→feature→velocity mapping at inference."""
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "encoder_version": ENCODER_VERSION,
            "feature_version": FEATURE_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "balance_hemispheres": BALANCE_HEMISPHERES,
            "gaze_params": gaze_params_dict(),
            "input_groups": [f"{t}-{s}" for t, s in INPUT_GROUPS],
            "input_channel_meaning": list(INPUT_CHANNEL_MEANING),
            "sample_ms": self.sample_ms,
            "feature_ms": self.feature_ms,
            "drive_hz_scale": self.drive_hz_scale,
            "response_seed": RESPONSE_RNG_SEED,
            "deadband": DEADBAND,
        }

    def training_settings(self) -> dict[str, Any]:
        """Fit hyperparameters recorded in metadata; not required to match on load."""
        return {
            "train_seed": TRAIN_RNG_SEED,
            "ridge": self.ridge,
            "n_train_random": self.n_train_random,
            "n_train_baseline": self.n_train_baseline,
            "n_holdout": self.n_holdout,
        }

    def _ensure_brain(self) -> FlyBrain:
        if self.brain is not None:
            return self.brain
        if not self.connectome_path.is_file():
            raise FileNotFoundError(f"connectome missing: {self.connectome_path}")
        self.graph_sha256 = sha256_file(self.connectome_path)
        brain = FlyBrain(
            str(self.connectome_path),
            GAZE_PARAMS,
            seed=0,
            balance_hemispheres=BALANCE_HEMISPHERES,
        )
        groups = []
        input_ids: list[list[int]] = []
        for typ, side in INPUT_GROUPS:
            idx = np.flatnonzero((brain.type == typ) & (brain.side == side))
            if idx.size == 0:
                raise RuntimeError(f"input group empty: {typ}-{side}")
            brain.driven[idx] = True
            groups.append(idx)
            input_ids.append(brain.bodyId[idx].astype(np.int64).tolist())
        brain._driven_idx = np.flatnonzero(brain.driven)
        dn = brain.pop["DN"]
        if dn.size == 0:
            raise RuntimeError("no descending neurons in connectome")
        self.brain = brain
        self._input_groups = groups
        self._input_ids = input_ids
        self._dn_idx = dn
        self._dn_ids = brain.bodyId[dn].astype(np.int64)
        self.model_info = {
            "neurons": int(brain.N),
            "edges": int(brain.n_edges),
            "fast_edges": int(brain.n_edges),
            "input_neurons": int(sum(g.size for g in groups)),
            "readout_neurons": int(dn.size),
            "kind": "connectome-reservoir",
        }
        return brain

    def _error_to_drive(self, dx: float, dy: float) -> list[float]:
        # [neg_x, pos_x, neg_y, pos_y] * scale — arbitrary engineered map.
        return [
            max(-dx, 0.0) * self.drive_hz_scale,
            max(dx, 0.0) * self.drive_hz_scale,
            max(-dy, 0.0) * self.drive_hz_scale,
            max(dy, 0.0) * self.drive_hz_scale,
        ]

    def neural_features(self, dx: float, dy: float) -> np.ndarray:
        """
        Run one trial-reset reservoir window and return mean DN voltage features.

        Resets transient neural state and RNG (seed RESPONSE_RNG_SEED) so the
        same (dx, dy) maps to a repeatable feature vector.
        """
        brain = self._ensure_brain()
        dx, dy = _validate_error(dx, dy)
        brain.reset()
        brain.rng = np.random.default_rng(RESPONSE_RNG_SEED)
        brain.drive_hz.fill(0.0)
        for group, hz in zip(self._input_groups, self._error_to_drive(dx, dy)):
            brain.drive_hz[group] = hz
        acc = np.zeros(self._dn_idx.size, dtype=np.float64)
        start = self.sample_ms - self.feature_ms
        for i in range(self.sample_ms):
            brain.step()
            if i >= start:
                acc += brain.v[self._dn_idx]
        return (acc / float(self.feature_ms)).astype(np.float64)

    def _decode(self, features: np.ndarray) -> tuple[float, float, int]:
        if self.scales is None or self.W is None:
            raise RuntimeError("controller untrained")
        xs = features / self.scales
        out = xs @ self.W
        vx, vy = _deadband_clip(float(out[0]), float(out[1]))
        active = int(np.count_nonzero(np.abs(features) > ACTIVE_FEATURE_STD))
        return vx, vy, active

    def predict(self, dx: float, dy: float) -> dict[str, Any]:
        """Map gaze error → velocity via DN reservoir features + linear readout."""
        with self._lock:
            if not self.ready or self.W is None or self.scales is None:
                raise RuntimeError("controller untrained")
            dx, dy = _validate_error(dx, dy)
            t0 = time.perf_counter()
            if dx == 0.0 and dy == 0.0:
                vx, vy, active = 0.0, 0.0, 0
            else:
                features = self.neural_features(dx, dy)
                vx, vy, active = self._decode(features)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "vx": vx,
                "vy": vy,
                "latency_ms": latency_ms,
                "active_dn": active,
                "neurons": int(self.brain.N) if self.brain is not None else 0,
                "ready": True,
            }

    def train(
        self,
        progress_callback: Callable[[float, str], None] | None = None,
        cancel_event: threading.Event | None = None,
        out_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Fit linear DN readout on trial-reset reservoir responses.

        Connectome stays fixed. Readout (scales/W) is swapped only after a
        successful finite fit that passes causality (+ holdout gate when
        enforce_acceptance=True). Failures leave any prior readout untouched.
        """
        with self._lock:
            brain = self._ensure_brain()
            assert self.graph_sha256 is not None

            def progress(p: float, phase: str) -> None:
                if progress_callback is not None:
                    progress_callback(float(np.clip(p, 0.0, 1.0)), phase)

            def cancelled() -> bool:
                return cancel_event is not None and cancel_event.is_set()

            progress(0.0, "sampling")
            rng = np.random.default_rng(TRAIN_RNG_SEED)
            errors = np.vstack(
                [
                    rng.uniform(-1.0, 1.0, (self.n_train_random, 2)),
                    np.zeros((self.n_train_baseline, 2)),
                ]
            )
            n_train = len(errors)
            X = np.zeros((n_train, self._dn_idx.size), dtype=np.float64)
            latencies: list[float] = []
            for i, err in enumerate(errors):
                if cancelled():
                    raise RuntimeError("training cancelled")
                t0 = time.perf_counter()
                X[i] = self.neural_features(float(err[0]), float(err[1]))
                latencies.append(time.perf_counter() - t0)
                if (i + 1) % 10 == 0 or i + 1 == n_train:
                    progress(0.55 * (i + 1) / n_train, "sampling")

            progress(0.58, "fitting")
            scales = X.std(axis=0)
            scales = np.where(scales < SCALE_FLOOR, 1.0, scales)
            Xs = X / scales
            # W: (n_features, 2); ridge on sample Gram — no intercept, no raw targets.
            gram = Xs @ Xs.T + self.ridge * np.eye(n_train)
            W = Xs.T @ np.linalg.solve(gram, errors)

            progress(0.62, "holdout")
            hold = rng.uniform(-1.0, 1.0, (self.n_holdout, 2))
            Xh = np.zeros((self.n_holdout, self._dn_idx.size), dtype=np.float64)
            for i, err in enumerate(hold):
                if cancelled():
                    raise RuntimeError("training cancelled")
                t0 = time.perf_counter()
                Xh[i] = self.neural_features(float(err[0]), float(err[1]))
                latencies.append(time.perf_counter() - t0)
                progress(0.62 + 0.30 * (i + 1) / self.n_holdout, "holdout")

            pred = (Xh / scales) @ W
            err = pred - hold
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            sign_acc = float(np.mean(np.sign(pred) == np.sign(hold)))

            zero_pred = np.zeros_like(hold)
            zero_mae = float(np.mean(np.abs(zero_pred - hold)))
            zero_rmse = float(np.sqrt(np.mean((zero_pred - hold) ** 2)))
            muted_pred = (np.zeros_like(Xh) / scales) @ W
            muted_mae = float(np.mean(np.abs(muted_pred - hold)))
            muted_norm = float(np.linalg.norm(muted_pred))

            # Neural causality: with synaptic propagation muted, features collapse.
            progress(0.95, "causality")
            saved_w = brain._out_w.copy()
            try:
                brain._out_w[:] = 0.0
                ablated = self.neural_features(0.7, -0.5)
            finally:
                brain._out_w[:] = saved_w
            live = self.neural_features(0.7, -0.5)
            causality_ok = bool(
                np.linalg.norm(live) > 1e-3
                and np.linalg.norm(ablated) < 0.05 * max(np.linalg.norm(live), 1e-9)
            )

            active_features = int((X.std(axis=0) > ACTIVE_FEATURE_STD).sum())
            nonzero_features = int(np.count_nonzero(np.abs(X).max(axis=0) > 0))
            changed_weights = int(np.count_nonzero(np.abs(W) > 1e-12))

            metrics = {
                "mae": mae,
                "rmse": rmse,
                "sign_accuracy": sign_acc,
                "zero_output_mae": zero_mae,
                "zero_output_rmse": zero_rmse,
                "muted_features_mae": muted_mae,
                "muted_features_pred_norm": muted_norm,
                "causality_ok": causality_ok,
                "ablated_feature_norm": float(np.linalg.norm(ablated)),
                "live_feature_norm": float(np.linalg.norm(live)),
                "n_train": n_train,
                "n_train_random": self.n_train_random,
                "n_train_baseline": self.n_train_baseline,
                "n_holdout": self.n_holdout,
                "train_seed": TRAIN_RNG_SEED,
                "response_seed": RESPONSE_RNG_SEED,
                "ridge": self.ridge,
                "sample_ms": self.sample_ms,
                "feature_ms": self.feature_ms,
                "drive_hz_scale": self.drive_hz_scale,
                "active_dn_features": active_features,
                "nonzero_dn_features": nonzero_features,
                "changed_readout_weights": changed_weights,
                "mean_latency_s": float(np.mean(latencies)) if latencies else 0.0,
                "neurons": int(brain.N),
                "fast_edges": int(brain.n_edges),
                "readout_neurons": int(self._dn_idx.size),
                "input_neurons": int(sum(g.size for g in self._input_groups)),
                "input_channel_meaning": list(INPUT_CHANNEL_MEANING),
                "enforce_acceptance": self.enforce_acceptance,
                "trial_reset": (
                    "Each sample resets membrane/synapse/delay state and RNG "
                    f"seed {RESPONSE_RNG_SEED}, integrates {self.sample_ms} ms, "
                    f"averages DN voltage over final {self.feature_ms} ms."
                ),
            }

            if not (np.isfinite(scales).all() and np.isfinite(W).all() and np.isfinite(pred).all()):
                raise RuntimeError("training failed: non-finite scales/W/predictions")
            if not causality_ok:
                raise RuntimeError(
                    "training failed: causality check failed "
                    f"(live_norm={metrics['live_feature_norm']}, "
                    f"ablated_norm={metrics['ablated_feature_norm']})"
                )
            if self.enforce_acceptance:
                if mae >= ACCEPT_MAE_MAX or sign_acc < ACCEPT_SIGN_MIN:
                    raise RuntimeError(
                        "training failed: holdout acceptance gate "
                        f"(mae={mae:.4f} need <{ACCEPT_MAE_MAX}, "
                        f"sign_accuracy={sign_acc:.4f} need >={ACCEPT_SIGN_MIN})"
                    )

            # Swap readout only after a complete successful fit.
            self.scales = scales.astype(np.float64)
            self.W = W.astype(np.float64)
            self.metrics = metrics
            self.ready = True
            self.load_error = None

            dest = Path(out_path) if out_path else self.checkpoint_path
            self.save_checkpoint(dest)
            progress(1.0, "done")
            return metrics

    def save_checkpoint(self, path: Path | str | None = None) -> Path:
        if self.W is None or self.scales is None or self.metrics is None:
            raise RuntimeError("nothing to save")
        self._ensure_brain()
        assert self.graph_sha256 is not None
        path = Path(path or self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            **self.inference_settings(),
            "graph_sha256": self.graph_sha256,
            "connectome_path": str(self.connectome_path),
            "input_ids": self._input_ids,
            "input_channel_note": (
                "Arbitrary engineered directional channels for reservoir drive; "
                "not claims about natural LC4/LPLC2 preferred directions."
            ),
            "trial_reset_reservoir": (
                f"reset+rng({RESPONSE_RNG_SEED}) each window; integrate {self.sample_ms}ms; "
                f"mean DN V over final {self.feature_ms}ms"
            ),
            "training": self.training_settings(),
            "model": self.model_info,
            "metrics": self.metrics,
        }

        fd, tmp_name = tempfile.mkstemp(prefix="gaze-controller-", suffix=".npz", dir=str(path.parent))
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            np.savez_compressed(
                tmp_path,
                meta_json=np.asarray(json.dumps(_jsonable(meta)), dtype=np.str_),
                scales=self.scales.astype(np.float64),
                W=self.W.astype(np.float64),
                dn_ids=self._dn_ids.astype(np.int64),
                dn_idx=self._dn_idx.astype(np.int64),
            )
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        self.checkpoint_path = path
        return path

    def _require_meta_equal(self, meta: dict[str, Any], key: str, expected: Any) -> None:
        got = _jsonable(meta.get(key))
        exp = _jsonable(expected)
        if got != exp:
            raise ValueError(f"{key} mismatch: got {got!r} expected {exp!r}")

    def load_checkpoint(self, path: Path | str | None = None) -> str:
        """
        Load readout checkpoint. Returns 'ready' or a clear untrained reason string.

        Strictly validates all inference-defining settings. Training hyperparams
        (n_train_*, ridge, train_seed) are recorded but not required to match the
        live controller instance.
        """
        path = Path(path or self.checkpoint_path)
        if not path.is_file():
            self.ready = False
            self.load_error = "checkpoint absent"
            return self.load_error
        try:
            data = np.load(path, allow_pickle=False)
            if "meta_json" not in data.files or "scales" not in data.files or "W" not in data.files:
                raise ValueError("missing required arrays")
            meta = json.loads(str(data["meta_json"].item()))

            brain = self._ensure_brain()
            assert self.graph_sha256 is not None
            expected = self.inference_settings()
            for key, exp in expected.items():
                self._require_meta_equal(meta, key, exp)
            if meta.get("graph_sha256") != self.graph_sha256:
                raise ValueError("graph_sha256 mismatch")

            input_ids = _jsonable(meta.get("input_ids"))
            if input_ids != _jsonable(self._input_ids):
                raise ValueError("input_ids mismatch")

            scales = np.asarray(data["scales"], dtype=np.float64)
            W = np.asarray(data["W"], dtype=np.float64)
            dn_ids = np.asarray(data["dn_ids"], dtype=np.int64)
            if scales.ndim != 1 or W.ndim != 2 or W.shape[1] != 2:
                raise ValueError("bad scales/W shape")
            if scales.shape[0] != W.shape[0]:
                raise ValueError("scales/W length mismatch")
            if scales.shape[0] != self._dn_idx.size:
                raise ValueError("readout width != DN count")
            if dn_ids.shape != self._dn_ids.shape or not np.array_equal(dn_ids, self._dn_ids):
                raise ValueError("DN id mismatch")
            if not np.isfinite(scales).all() or not np.isfinite(W).all():
                raise ValueError("non-finite scales/W")
            if (scales <= 0).any():
                raise ValueError("non-positive scales")

            self.scales = scales
            self.W = W
            self.metrics = meta.get("metrics")
            if self.model_info is None:
                self.model_info = meta.get("model")
            self.ready = True
            self.load_error = None
            self.checkpoint_path = path
            return "ready"
        except Exception as exc:  # noqa: BLE001 — surface any corrupt/invalid reason
            self.ready = False
            self.scales = None
            self.W = None
            self.metrics = None
            self.load_error = f"checkpoint corrupt/invalid: {exc}"
            return self.load_error

    def info(self) -> dict[str, Any]:
        brain_loaded = self.brain is not None
        if not brain_loaded:
            try:
                self._ensure_brain()
            except Exception as exc:  # noqa: BLE001
                return {"error": str(exc), "ready": False}
        assert self.brain is not None
        return {
            "ready": self.ready,
            "load_error": self.load_error,
            "connectome": str(self.connectome_path),
            "graph_sha256": self.graph_sha256,
            "upstream_commit": UPSTREAM_COMMIT,
            "checkpoint": str(self.checkpoint_path),
            "model": self.model_info,
            "metrics": self.metrics,
            "inference": self.inference_settings(),
            "training": self.training_settings(),
            "encoder_version": ENCODER_VERSION,
            "feature_version": FEATURE_VERSION,
            "input_channel_meaning": list(INPUT_CHANNEL_MEANING),
            "trial_reset": (
                f"reset+rng({RESPONSE_RNG_SEED}); {self.sample_ms}ms integrate; "
                f"final {self.feature_ms}ms DN V mean"
            ),
        }


def cmd_train(args: argparse.Namespace) -> int:
    ctrl = GazeController(
        connectome_path=args.connectome,
        checkpoint_path=args.out,
        auto_load_brain=True,
        enforce_acceptance=True,
    )

    def on_progress(p: float, phase: str) -> None:
        print(f"[gaze-train] {phase} {p:.0%}", flush=True)

    metrics = ctrl.train(progress_callback=on_progress, out_path=args.out)
    print(json.dumps({"saved": str(args.out), "metrics": metrics}, indent=2, sort_keys=True))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    ctrl = GazeController(
        connectome_path=args.connectome,
        checkpoint_path=args.checkpoint,
        auto_load_brain=True,
    )
    print(json.dumps(ctrl.info(), indent=2, sort_keys=True, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fly gaze connectome-reservoir controller")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Fit DN readout and save checkpoint")
    tr.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    tr.add_argument("--connectome", type=Path, default=DEFAULT_CONNECTOME)
    tr.set_defaults(func=cmd_train)

    info = sub.add_parser("info", help="Show controller / checkpoint status")
    info.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    info.add_argument("--connectome", type=Path, default=DEFAULT_CONNECTOME)
    info.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
