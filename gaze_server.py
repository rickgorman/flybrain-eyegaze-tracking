#!/usr/bin/env python3
"""
Local gaze controller HTTP server.

Serves ONLY the web/ static tree (including vendor assets) and a small JSON API
that drives brain_controller.GazeController. Never accepts webcam pixels or
exposes workspace/data/captures.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from brain_controller import DEFAULT_CHECKPOINT, DEFAULT_CONNECTOME, GazeController

PROJECT_ROOT = Path(__file__).resolve().parent
WEB_ROOT = PROJECT_ROOT / "web"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
MAX_BODY = 16 * 1024
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Content types for MediaPipe / wasm assets.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".task")


def require_loopback_host(host: str) -> str:
    h = (host or "").strip().lower()
    if h not in LOOPBACK_HOSTS:
        raise SystemExit(
            f"refusing non-loopback --host {host!r}; bind only 127.0.0.1 / localhost / ::1"
        )
    return h


def is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def np_isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


class GazeApp:
    def __init__(
        self,
        connectome_path: Path,
        checkpoint_path: Path,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self.host = require_loopback_host(host)
        self.port = port
        self.connectome_path = connectome_path
        self.checkpoint_path = checkpoint_path

        self.controller = GazeController(
            connectome_path=connectome_path,
            checkpoint_path=checkpoint_path,
            auto_load_brain=True,
            enforce_acceptance=True,
        )
        self._op_lock = threading.Lock()
        self._train_lock = threading.Lock()
        self.training = False
        self.progress = 0.0
        self.phase = "idle"
        self.error: str | None = None
        self._cancel = threading.Event()
        self._train_thread: threading.Thread | None = None

    def status(self) -> dict[str, Any]:
        ctrl = self.controller
        error = self.error
        if error is None and not ctrl.ready:
            error = ctrl.load_error
        return {
            "ready": bool(ctrl.ready) and not self.training,
            "training": self.training,
            "progress": float(self.progress),
            "phase": self.phase,
            "error": error,
            "model": ctrl.model_info
            or {
                "neurons": 0,
                "edges": 0,
                "fast_edges": 0,
                "input_neurons": 0,
                "readout_neurons": 0,
                "kind": "connectome-reservoir",
            },
            "metrics": None if self.training else ctrl.metrics,
        }

    def start_train(self) -> tuple[int, dict[str, Any]]:
        if not self._train_lock.acquire(blocking=False):
            return 409, {"accepted": False, "error": "already training"}
        if self.training:
            self._train_lock.release()
            return 409, {"accepted": False, "error": "already training"}
        self.training = True
        self.progress = 0.0
        self.phase = "starting"
        self.error = None
        self._cancel.clear()

        def runner() -> None:
            try:
                # Fresh instance so a failed fit cannot corrupt the live readout.
                trainer = GazeController(
                    connectome_path=self.connectome_path,
                    checkpoint_path=self.checkpoint_path,
                    auto_load_brain=True,
                    enforce_acceptance=True,
                )

                def on_progress(p: float, phase: str) -> None:
                    self.progress = float(p)
                    self.phase = phase

                metrics = trainer.train(
                    progress_callback=on_progress,
                    cancel_event=self._cancel,
                    out_path=self.checkpoint_path,
                )
                # Swap only after successful complete train + atomic save.
                with self._op_lock:
                    self.controller = trainer
                self.progress = 1.0
                self.phase = "done"
                self.error = None
                print(
                    f"[gaze] train done mae={metrics.get('mae')} "
                    f"sign={metrics.get('sign_accuracy')}"
                )
            except Exception as exc:  # noqa: BLE001
                self.error = str(exc)
                self.phase = "error"
                print(f"[gaze] train failed: {exc}")
            finally:
                self.training = False
                self._train_lock.release()

        self._train_thread = threading.Thread(target=runner, name="gaze-train", daemon=True)
        self._train_thread.start()
        return 202, {"accepted": True}

    def step(self, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        if self.training:
            return 409, {"error": "training", "ready": False}
        if not self.controller.ready:
            return 409, {
                "error": "untrained",
                "ready": False,
                "detail": self.controller.load_error,
            }
        try:
            dx = body["dx"]
            dy = body["dy"]
            seq = body["seq"]
        except KeyError as exc:
            return 400, {"error": f"missing field: {exc.args[0]}"}
        if isinstance(seq, bool) or not isinstance(seq, int):
            return 400, {"error": "seq must be an integer"}
        if seq < 0:
            return 400, {"error": "seq must be non-negative"}
        if not is_json_number(dx) or not is_json_number(dy):
            return 400, {"error": "dx/dy must be JSON numbers"}
        dx_f = float(dx)
        dy_f = float(dy)
        if not (np_isfinite(dx_f) and np_isfinite(dy_f)):
            return 400, {"error": "dx/dy must be finite"}
        if not (abs(dx_f) <= 1.0 and abs(dy_f) <= 1.0):
            return 400, {"error": "dx/dy out of bounds"}

        if not self._op_lock.acquire(blocking=False):
            return 409, {"error": "busy", "ready": True}
        try:
            out = self.controller.predict(dx_f, dy_f)
            return 200, {
                "seq": seq,
                "vx": out["vx"],
                "vy": out["vy"],
                "latency_ms": out["latency_ms"],
                "active_dn": out["active_dn"],
                "neurons": out["neurons"],
                "ready": True,
            }
        except ValueError as exc:
            return 400, {"error": str(exc)}
        except RuntimeError as exc:
            return 409, {"error": str(exc), "ready": False}
        finally:
            self._op_lock.release()

    def pause(self) -> tuple[int, dict[str, Any]]:
        # Harmless ack; optionally clear drive without mutating readout.
        if self._op_lock.acquire(blocking=False):
            try:
                brain = self.controller.brain
                if brain is not None:
                    brain.drive_hz.fill(0.0)
            finally:
                self._op_lock.release()
        return 200, {"ok": True}


def make_handler(app: GazeApp) -> type[BaseHTTPRequestHandler]:
    host = app.host.lower()
    allowed_hosts = {
        f"{host}:{app.port}",
        f"localhost:{app.port}",
        f"127.0.0.1:{app.port}",
    }
    if host == "::1":
        allowed_hosts.add(f"[::1]:{app.port}")
    if app.port == 80:
        allowed_hosts.update({host, "localhost", "127.0.0.1"})

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            try:
                msg = fmt % args
            except Exception:  # noqa: BLE001
                msg = str(fmt)
            # ~8 Hz step traffic is too chatty; keep errors and other routes.
            if '"POST /api/step ' in msg and '" 200 ' in msg:
                return
            print(f"[gaze] {self.address_string()} {msg}")

        def _origin_ok(self, origin: str) -> bool:
            try:
                parsed = urlparse(origin)
            except Exception:  # noqa: BLE001
                return False
            if parsed.scheme not in ("http", "https"):
                return False
            if parsed.path not in ("",):
                return False
            if parsed.params or parsed.query or parsed.fragment:
                return False
            origin_host = parsed.netloc.lower()
            return origin_host in allowed_hosts

        def _local_ok(self) -> bool:
            host_hdr = (self.headers.get("Host") or "").strip().lower()
            if host_hdr not in allowed_hosts:
                return False
            origin = (self.headers.get("Origin") or "").strip()
            if origin and not self._origin_ok(origin):
                return False
            return True

        def _json(self, code: int, obj: dict[str, Any], *, close: bool = False) -> None:
            raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            if close:
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(raw)

        def _drain_body(self) -> None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            left = max(0, min(length, 1 << 20))
            while left > 0:
                chunk = self.rfile.read(min(left, 65536))
                if not chunk:
                    break
                left -= len(chunk)

        def _read_json(self) -> tuple[dict[str, Any] | None, str | None]:
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype != "application/json":
                self._drain_body()
                return None, "Content-Type must be application/json"
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None, "bad Content-Length"
            if length < 0 or length > MAX_BODY:
                self._drain_body()
                return None, "body too large"
            raw = self.rfile.read(length) if length else b"{}"
            if len(raw) > MAX_BODY:
                return None, "body too large"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except (UnicodeDecodeError, json.JSONDecodeError):
                return None, "invalid JSON"
            if not isinstance(data, dict):
                return None, "JSON object required"
            return data, None

        def do_GET(self) -> None:  # noqa: N802
            if not self._local_ok():
                self._json(403, {"error": "forbidden host/origin"}, close=True)
                return
            path = urlparse(self.path).path
            if path == "/api/status":
                self._json(200, app.status())
                return
            self._serve_static(path)

        def do_POST(self) -> None:  # noqa: N802
            body, err = self._read_json()
            if not self._local_ok():
                self._json(403, {"error": "forbidden host/origin"}, close=True)
                return
            if err is not None:
                self._json(400, {"error": err}, close=True)
                return
            assert body is not None
            path = urlparse(self.path).path
            if path == "/api/train":
                code, obj = app.start_train()
                self._json(code, obj)
                return
            if path == "/api/step":
                code, obj = app.step(body)
                self._json(code, obj, close=(code >= 400))
                return
            if path == "/api/pause":
                code, obj = app.pause()
                self._json(code, obj)
                return
            self._json(404, {"error": "not found"}, close=True)

        def _serve_static(self, path: str) -> None:
            if path in ("", "/"):
                path = "/index.html"
            rel = unquote(path).lstrip("/")
            if ".." in rel.split("/") or rel.startswith("/") or "\\" in rel:
                self._json(404, {"error": "not found"})
                return
            full = (WEB_ROOT / rel).resolve()
            try:
                rel_path = full.relative_to(WEB_ROOT.resolve())
            except ValueError:
                self._json(404, {"error": "not found"})
                return
            if not full.is_file():
                self.send_error(404, "File not found")
                return
            data = full.read_bytes()
            ctype = mimetypes.guess_type(str(full))[0] or "application/octet-stream"
            under_vendor = bool(rel_path.parts) and rel_path.parts[0] == "vendor"
            # Cache only vendor/ immutable assets. App JS/CSS must be no-store.
            if under_vendor and full.suffix in {".mjs", ".js", ".wasm", ".task", ".css"}:
                cache = "public, max-age=3600"
            else:
                cache = "no-store"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(data)

    return Handler


def run_server(host: str, port: int, connectome: Path, checkpoint: Path) -> None:
    if not WEB_ROOT.is_dir():
        raise SystemExit(f"missing web root: {WEB_ROOT}")
    host = require_loopback_host(host)
    app = GazeApp(connectome, checkpoint, host=host, port=port)
    handler = make_handler(app)
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    print(
        f"[gaze] http://{host}:{port}  checkpoint={checkpoint} "
        f"ready={app.controller.ready} error={app.status().get('error')}"
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("[gaze] shutting down")
    finally:
        server.shutdown()
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local fly gaze server")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--connectome", type=Path, default=DEFAULT_CONNECTOME)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_server(args.host, args.port, args.connectome, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
