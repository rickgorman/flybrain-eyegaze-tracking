"""Unit tests for brain_controller — synthetic tiny graph only."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

import brain_controller as bc


def _write_tiny_gaze_connectome(path: Path) -> Path:
    """
    Minimal graph with LC4/LPLC2 inputs and DN readouts.

    Strong feedforward edges so trial-reset DN voltages respond to drive.
    """
    n = 11
    bodyId = np.arange(1000, 1000 + n, dtype=np.int64)
    typ = np.array(
        ["LC4", "LC4", "LPLC2", "LPLC2"] + [f"DN_test_{i}" for i in range(6)] + ["X"],
        dtype=str,
    )
    side = np.array(["L", "R", "L", "R"] + ["M"] * 7, dtype=str)
    cls = np.array(["visual"] * 4 + ["DN"] * 6 + ["other"], dtype=str)
    sc = np.array(["visual_projection"] * 4 + ["descending_neuron"] * 6 + ["central"], dtype=str)
    nt = np.array(["acetylcholine"] * n, dtype=str)
    sign = np.ones(n, dtype=np.float32)

    pre: list[int] = []
    post: list[int] = []
    w: list[float] = []

    def edge(a: int, b: int, weight: float = 80.0) -> None:
        pre.append(a)
        post.append(b)
        w.append(weight)

    for src, dns in ((0, (4, 5)), (1, (6, 7)), (2, (8, 4)), (3, (9, 5))):
        for dn in dns:
            edge(src, dn, 120.0)

    np.savez(
        path,
        bodyId=bodyId,
        type=typ,
        cls=cls,
        sc=sc,
        nt=nt,
        side=side,
        sign=sign,
        pre=np.asarray(pre, dtype=np.int64),
        post=np.asarray(post, dtype=np.int64),
        w=np.asarray(w, dtype=np.float32),
    )
    return path


@pytest.fixture
def tiny_gaze_connectome(tmp_path: Path) -> Path:
    return _write_tiny_gaze_connectome(tmp_path / "tiny-gaze.npz")


@pytest.fixture
def tiny_controller(tiny_gaze_connectome: Path, tmp_path: Path) -> bc.GazeController:
    return bc.GazeController(
        connectome_path=tiny_gaze_connectome,
        checkpoint_path=tmp_path / "gaze-controller.npz",
        n_train_random=24,
        n_train_baseline=2,
        n_holdout=8,
        sample_ms=40,
        feature_ms=20,
        enforce_acceptance=False,  # tiny fixture is not the production holdout gate
        auto_load_brain=True,
    )


def test_arg_validation(tiny_gaze_connectome: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        bc.GazeController(
            connectome_path=tiny_gaze_connectome,
            checkpoint_path=tmp_path / "x.npz",
            sample_ms=0,
            feature_ms=1,
            enforce_acceptance=False,
            auto_load_brain=False,
        )
    with pytest.raises(ValueError):
        bc.GazeController(
            connectome_path=tiny_gaze_connectome,
            checkpoint_path=tmp_path / "x.npz",
            sample_ms=10,
            feature_ms=20,
            enforce_acceptance=False,
            auto_load_brain=False,
        )
    with pytest.raises(ValueError):
        bc.GazeController(
            connectome_path=tiny_gaze_connectome,
            checkpoint_path=tmp_path / "x.npz",
            n_train_random=0,
            n_train_baseline=0,
            enforce_acceptance=False,
            auto_load_brain=False,
        )
    with pytest.raises(ValueError):
        bc.GazeController(
            connectome_path=tiny_gaze_connectome,
            checkpoint_path=tmp_path / "x.npz",
            ridge=0.0,
            enforce_acceptance=False,
            auto_load_brain=False,
        )
    with pytest.raises(ValueError):
        bc.GazeController(
            connectome_path=tiny_gaze_connectome,
            checkpoint_path=tmp_path / "x.npz",
            drive_hz_scale=float("nan"),
            enforce_acceptance=False,
            auto_load_brain=False,
        )


def test_untrained_predict_rejects(tiny_controller: bc.GazeController) -> None:
    assert tiny_controller.ready is False
    with pytest.raises(RuntimeError, match="untrained"):
        tiny_controller.predict(0.1, -0.2)


def test_invalid_error_vectors(tiny_controller: bc.GazeController) -> None:
    with pytest.raises(ValueError):
        tiny_controller.neural_features(float("nan"), 0.0)
    with pytest.raises(ValueError):
        tiny_controller.neural_features(1.5, 0.0)


def test_train_predict_zero_and_deadband(tiny_controller: bc.GazeController, tmp_path: Path) -> None:
    metrics = tiny_controller.train(out_path=tmp_path / "ckpt.npz")
    assert tiny_controller.ready is True
    assert "sign_accuracy" in metrics
    assert "causality_ok" in metrics
    assert metrics["muted_features_pred_norm"] == pytest.approx(0.0, abs=1e-9)

    z = tiny_controller.predict(0.0, 0.0)
    assert z["vx"] == 0.0 and z["vy"] == 0.0

    out = tiny_controller.predict(0.8, -0.6)
    assert -1.0 <= out["vx"] <= 1.0
    assert -1.0 <= out["vy"] <= 1.0
    assert out["ready"] is True


def test_checkpoint_roundtrip_and_corrupt(tiny_controller: bc.GazeController, tmp_path: Path) -> None:
    dest = tmp_path / "ckpt.npz"
    tiny_controller.train(out_path=dest)
    ctrl2 = bc.GazeController(
        connectome_path=tiny_controller.connectome_path,
        checkpoint_path=dest,
        sample_ms=40,
        feature_ms=20,
        enforce_acceptance=False,
        auto_load_brain=True,
    )
    assert ctrl2.ready is True
    a = tiny_controller.predict(0.4, 0.2)
    b = ctrl2.predict(0.4, 0.2)
    assert a["vx"] == pytest.approx(b["vx"], abs=1e-9)
    assert a["vy"] == pytest.approx(b["vy"], abs=1e-9)

    bad = tmp_path / "bad.npz"
    np.savez(bad, scales=np.array([1.0]), W=np.array([[1.0, 2.0]]))
    status = ctrl2.load_checkpoint(bad)
    assert status != "ready"
    assert ctrl2.ready is False


def _rewrite_meta(src: Path, dest: Path, mutator) -> None:
    data = np.load(src, allow_pickle=False)
    meta = json.loads(str(data["meta_json"].item()))
    mutator(meta)
    np.savez_compressed(
        dest,
        meta_json=np.asarray(json.dumps(meta), dtype=np.str_),
        scales=data["scales"],
        W=data["W"],
        dn_ids=data["dn_ids"],
        dn_idx=data["dn_idx"],
    )


def test_wrong_graph_hash_rejected(tiny_controller: bc.GazeController, tmp_path: Path) -> None:
    dest = tmp_path / "ckpt.npz"
    tiny_controller.train(out_path=dest)
    other = tmp_path / "wrong-hash.npz"
    _rewrite_meta(dest, other, lambda m: m.__setitem__("graph_sha256", "0" * 64))
    status = tiny_controller.load_checkpoint(other)
    assert "graph_sha256" in status
    assert tiny_controller.ready is False


def test_inference_setting_tamper_rejected(tiny_controller: bc.GazeController, tmp_path: Path) -> None:
    dest = tmp_path / "ckpt.npz"
    tiny_controller.train(out_path=dest)

    cases = [
        ("sample_ms", lambda m: m.__setitem__("sample_ms", 99)),
        ("feature_ms", lambda m: m.__setitem__("feature_ms", 1)),
        ("drive_hz_scale", lambda m: m.__setitem__("drive_hz_scale", 999.0)),
        ("response_seed", lambda m: m.__setitem__("response_seed", 0)),
        ("upstream_commit", lambda m: m.__setitem__("upstream_commit", "deadbeef")),
        ("input_groups", lambda m: m.__setitem__("input_groups", ["X-L", "X-R", "Y-L", "Y-R"])),
        ("gaze_params", lambda m: m["gaze_params"].__setitem__("noise", 0.5)),
        ("deadband", lambda m: m.__setitem__("deadband", 0.5)),
        ("checkpoint_version", lambda m: m.__setitem__("checkpoint_version", 1)),
    ]
    for label, mut in cases:
        path = tmp_path / f"tamper-{label}.npz"
        _rewrite_meta(dest, path, mut)
        status = tiny_controller.load_checkpoint(path)
        assert status != "ready", label
        assert tiny_controller.ready is False

    # Training hyperparams may differ from the live instance and must still load.
    ok = tmp_path / "train-hyper-ok.npz"
    _rewrite_meta(
        dest,
        ok,
        lambda m: (
            m.setdefault("training", {}).__setitem__("n_train_random", 999),
            m["training"].__setitem__("ridge", 99.0),
        ),
    )
    # Reload with different live training args — inference settings still match.
    ctrl = bc.GazeController(
        connectome_path=tiny_controller.connectome_path,
        checkpoint_path=ok,
        n_train_random=1,
        n_train_baseline=1,
        n_holdout=2,
        ridge=2.0,
        sample_ms=40,
        feature_ms=20,
        enforce_acceptance=False,
        auto_load_brain=True,
    )
    assert ctrl.ready is True


def test_failed_train_keeps_old_readout(tiny_controller: bc.GazeController, tmp_path: Path) -> None:
    dest = tmp_path / "ckpt.npz"
    tiny_controller.train(out_path=dest)
    assert tiny_controller.ready is True
    before = tiny_controller.predict(0.3, -0.2)
    old_W = tiny_controller.W.copy()

    # Force acceptance gate failure path without destroying prior readout.
    tiny_controller.enforce_acceptance = True
    # Monkeypatch holdout metrics by temporarily raising after fit via causality sabotage:
    # zeroing outgoing weights during train's causality window is internal; instead
    # stub neural_features after first successful train by making causality fail.
    real_features = tiny_controller.neural_features

    calls = {"n": 0}

    def flaky(dx, dy):
        # Let sampling/holdout run; during causality live/ablated calls after holdout,
        # return zeros so causality_ok becomes false.
        out = real_features(dx, dy)
        calls["n"] += 1
        # After train samples (26) + holdout (8) = 36, causality does 2 more calls.
        if calls["n"] > tiny_controller.n_train_random + tiny_controller.n_train_baseline + tiny_controller.n_holdout:
            return np.zeros_like(out)
        return out

    tiny_controller.neural_features = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="causality"):
        tiny_controller.train(out_path=tmp_path / "fail.npz")
    tiny_controller.neural_features = real_features  # type: ignore[method-assign]

    assert tiny_controller.ready is True
    assert np.array_equal(tiny_controller.W, old_W)
    after = tiny_controller.predict(0.3, -0.2)
    assert after["vx"] == pytest.approx(before["vx"], abs=1e-9)
    assert after["vy"] == pytest.approx(before["vy"], abs=1e-9)
    assert not (tmp_path / "fail.npz").exists()


def test_neural_causality_ablation(tiny_controller: bc.GazeController) -> None:
    tiny_controller.train()
    live = tiny_controller.neural_features(0.9, -0.7)
    brain = tiny_controller.brain
    assert brain is not None
    saved = brain._out_w.copy()
    try:
        brain._out_w[:] = 0.0
        ablated = tiny_controller.neural_features(0.9, -0.7)
    finally:
        brain._out_w[:] = saved
    assert np.linalg.norm(live) > np.linalg.norm(ablated)
    assert float(np.linalg.norm(ablated)) < 0.05 * float(np.linalg.norm(live)) + 1e-9


def test_train_cancel(tiny_controller: bc.GazeController) -> None:
    ev = threading.Event()
    ev.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        tiny_controller.train(cancel_event=ev)


def test_metadata_states_arbitrary_channels(tiny_controller: bc.GazeController, tmp_path: Path) -> None:
    dest = tmp_path / "ckpt.npz"
    tiny_controller.train(out_path=dest)
    meta = json.loads(str(np.load(dest, allow_pickle=False)["meta_json"].item()))
    assert meta["checkpoint_version"] == bc.CHECKPOINT_VERSION
    assert meta["input_channel_meaning"] == list(bc.INPUT_CHANNEL_MEANING)
    assert "gaze_params" in meta and "noise" in meta["gaze_params"]
    assert "training" in meta and "ridge" in meta["training"]
    assert "input_ids" in meta
    assert "not claims" in meta["input_channel_note"].lower() or "Arbitrary" in meta["input_channel_note"]
    assert "reset" in meta["trial_reset_reservoir"].lower()
