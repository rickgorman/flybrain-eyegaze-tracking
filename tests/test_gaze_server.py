"""HTTP API tests for gaze_server — synthetic tiny graph only."""
from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import brain_controller as bc
import gaze_server
from tests.test_controller import _write_tiny_gaze_connectome


@pytest.fixture
def tiny_gaze_connectome(tmp_path: Path) -> Path:
    return _write_tiny_gaze_connectome(tmp_path / "tiny-gaze.npz")


@pytest.fixture
def trained_app(tiny_gaze_connectome: Path, tmp_path: Path) -> gaze_server.GazeApp:
    ckpt = tmp_path / "gaze-controller.npz"
    ctrl = bc.GazeController(
        connectome_path=tiny_gaze_connectome,
        checkpoint_path=ckpt,
        n_train_random=24,
        n_train_baseline=2,
        n_holdout=8,
        sample_ms=40,
        feature_ms=20,
        enforce_acceptance=False,
        auto_load_brain=True,
    )
    ctrl.train(out_path=ckpt)
    app = gaze_server.GazeApp(tiny_gaze_connectome, ckpt, host="127.0.0.1", port=0)
    app.controller = ctrl
    return app


@pytest.fixture
def live_server(trained_app: gaze_server.GazeApp):
    handler = gaze_server.make_handler(trained_app)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    port = server.server_address[1]
    trained_app.port = port
    handler = gaze_server.make_handler(trained_app)
    server.RequestHandlerClass = handler
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield trained_app, port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(
    port: int,
    method: str,
    path: str,
    body: dict | None = None,
    *,
    host: str = "127.0.0.1",
    origin: str | None = None,
    content_type: str | None = "application/json",
    raw: bytes | None = None,
) -> tuple[int, dict | str, dict[str, str]]:
    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": f"{host}:{port}"}
    if origin is not None:
        headers["Origin"] = origin
    payload = raw
    if body is not None:
        payload = json.dumps(body).encode()
        if content_type:
            headers["Content-Type"] = content_type
    elif method == "POST":
        payload = b"{}"
        if content_type:
            headers["Content-Type"] = content_type
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    resp_headers = {k.lower(): v for k, v in resp.getheaders()}
    conn.close()
    ctype = resp.getheader("Content-Type") or ""
    if "application/json" in ctype:
        return resp.status, json.loads(data.decode()), resp_headers
    return resp.status, data.decode("utf-8", errors="replace"), resp_headers


def test_status_and_step(live_server) -> None:
    app, port = live_server
    code, status, _ = _request(port, "GET", "/api/status")
    assert code == 200
    assert status["ready"] is True
    assert status["training"] is False
    assert status["error"] is None
    assert status["model"]["kind"] == "connectome-reservoir"

    code, out, _ = _request(port, "POST", "/api/step", {"dx": 0.5, "dy": -0.25, "seq": 3})
    assert code == 200
    assert out["seq"] == 3
    assert out["ready"] is True
    assert "vx" in out and "vy" in out
    assert "latency_ms" in out

    code, z, _ = _request(port, "POST", "/api/step", {"dx": 0.0, "dy": 0.0, "seq": 4})
    assert code == 200
    assert z["vx"] == 0.0 and z["vy"] == 0.0


def test_invalid_step_types_and_seq(live_server) -> None:
    _, port = live_server
    code, err, hdrs = _request(port, "POST", "/api/step", {"dx": 2.0, "dy": 0.0, "seq": 1})
    assert code == 400
    assert hdrs.get("connection") == "close"
    code, err, _ = _request(port, "POST", "/api/step", {"dx": 0.0, "dy": 0.0, "seq": 1.5})
    assert code == 400
    code, err, _ = _request(port, "POST", "/api/step", {"dx": "x", "dy": 0.0, "seq": 1})
    assert code == 400
    code, err, _ = _request(port, "POST", "/api/step", {"dx": True, "dy": 0.0, "seq": 1})
    assert code == 400
    assert "JSON numbers" in err["error"]
    code, err, _ = _request(port, "POST", "/api/step", {"dx": 0.0, "dy": False, "seq": 1})
    assert code == 400
    code, err, _ = _request(port, "POST", "/api/step", {"dx": 0.0, "dy": 0.0, "seq": -1})
    assert code == 400
    assert "non-negative" in err["error"]


def test_untrained_status_surfaces_load_error(tiny_gaze_connectome: Path, tmp_path: Path) -> None:
    ckpt = tmp_path / "missing.npz"
    app = gaze_server.GazeApp(tiny_gaze_connectome, ckpt, host="127.0.0.1", port=0)
    handler = gaze_server.make_handler(app)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    port2 = server.server_address[1]
    app.port = port2
    server.RequestHandlerClass = gaze_server.make_handler(app)
    t = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    try:
        code, status, _ = _request(port2, "GET", "/api/status")
        assert code == 200
        assert status["ready"] is False
        assert status["error"] is not None
        assert "checkpoint" in status["error"]
        code, err, _ = _request(port2, "POST", "/api/step", {"dx": 0.1, "dy": 0.1, "seq": 1})
        assert code == 409
        assert err["error"] == "untrained"
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=2)


def test_busy_step_returns_409(live_server) -> None:
    app, port = live_server
    held = threading.Event()
    release = threading.Event()

    def blocker() -> None:
        assert app._op_lock.acquire(blocking=True)
        held.set()
        release.wait(timeout=2)
        app._op_lock.release()

    th = threading.Thread(target=blocker, daemon=True)
    th.start()
    assert held.wait(timeout=1)
    try:
        code, err, _ = _request(port, "POST", "/api/step", {"dx": 0.2, "dy": 0.1, "seq": 9})
        assert code == 409
        assert err["error"] == "busy"
    finally:
        release.set()
        th.join(timeout=2)


def test_foreign_host_and_origin_rejected(live_server) -> None:
    _, port = live_server
    code, err, hdrs = _request(port, "GET", "/api/status", host="evil.example")
    assert code == 403
    assert hdrs.get("connection") == "close"
    code, err, hdrs = _request(
        port,
        "POST",
        "/api/step",
        {"dx": 0.0, "dy": 0.0, "seq": 1},
        origin="http://evil.example:8766",
    )
    assert code == 403
    assert hdrs.get("connection") == "close"
    # Origin with non-empty path must be rejected.
    code, err, _ = _request(
        port,
        "POST",
        "/api/step",
        {"dx": 0.0, "dy": 0.0, "seq": 1},
        origin=f"http://127.0.0.1:{port}/extra",
    )
    assert code == 403
    code, err, _ = _request(
        port,
        "POST",
        "/api/step",
        {"dx": 0.0, "dy": 0.0, "seq": 1},
        origin=f"ftp://127.0.0.1:{port}",
    )
    assert code == 403


def test_static_cache_policy(live_server) -> None:
    _, port = live_server
    code, _, hdrs = _request(port, "GET", "/")
    assert code == 200
    assert hdrs.get("cache-control") == "no-store"

    conn = HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/vendor/vision_bundle.mjs", headers={"Host": f"127.0.0.1:{port}"})
    resp = conn.getresponse()
    data = resp.read()
    cache = resp.getheader("Cache-Control")
    conn.close()
    assert resp.status == 200
    assert "javascript" in (resp.getheader("Content-Type") or "")
    assert cache == "public, max-age=3600"
    assert len(data) > 0

    code, _, _ = _request(port, "GET", "/../gaze_server.py")
    assert code in (404, 403)


def test_train_conflict(live_server) -> None:
    app, port = live_server
    app.training = True
    app.phase = "sampling"
    try:
        code, err, _ = _request(port, "POST", "/api/train", {})
        assert code == 409
        code, err, _ = _request(port, "POST", "/api/step", {"dx": 0.1, "dy": 0.1, "seq": 1})
        assert code == 409
        assert err["error"] == "training"
    finally:
        app.training = False


def test_pause_ack(live_server) -> None:
    _, port = live_server
    code, out, _ = _request(port, "POST", "/api/pause", {})
    assert code == 200
    assert out.get("ok") is True


def test_bad_content_type_and_oversize(live_server) -> None:
    _, port = live_server
    code, err, hdrs = _request(
        port,
        "POST",
        "/api/step",
        {"dx": 0.0, "dy": 0.0, "seq": 1},
        content_type="text/plain",
    )
    assert code == 400
    assert hdrs.get("connection") == "close"
    huge = b'{"dx":0,"dy":0,"seq":1,"pad":"' + (b"x" * (17 * 1024)) + b'"}'
    code, err, hdrs = _request(port, "POST", "/api/step", raw=huge, content_type="application/json")
    assert code == 400
    assert hdrs.get("connection") == "close"


def test_refuse_non_loopback_host() -> None:
    with pytest.raises(SystemExit):
        gaze_server.require_loopback_host("0.0.0.0")
    with pytest.raises(SystemExit):
        gaze_server.require_loopback_host("192.168.1.1")
    assert gaze_server.require_loopback_host("127.0.0.1") == "127.0.0.1"
    assert gaze_server.require_loopback_host("localhost") == "localhost"
