from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

MAX_EVENTS = 8000

_events_lock = threading.Lock()
_events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENTS)

_server: ThreadingHTTPServer | None = None
_server_thread: threading.Thread | None = None
_started_port: int | None = None


class BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_json(self, status: int, body: dict | list) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            with _events_lock:
                n = len(_events)
            self._send_json(200, {"ok": True, "buffered_events": n})
            return
        if parsed.path == "/events":
            qs = parse_qs(parsed.query)
            try:
                limit = int(qs.get("limit", ["500"])[0])
            except ValueError:
                limit = 500
            limit = max(1, min(limit, MAX_EVENTS))
            with _events_lock:
                chunk = list(_events)[-limit:]
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(chunk, ensure_ascii=False).encode("utf-8"))
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/ingest":
            self.send_response(404)
            self.end_headers()
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_response(400)
            self.end_headers()
            return
        if not isinstance(data, dict):
            self.send_response(400)
            self.end_headers()
            return
        with _events_lock:
            _events.append(data)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()


def start_bridge_server(host: str = "127.0.0.1", port: int = 8765) -> tuple[bool, int]:
    """
    Inicia o servidor de ingestão em thread daemon.
    Retorna (started_here, port): started_here=False se a porta já estiver ocupada
    (outro processo — ex.: outra aba Streamlit — já escuta).
    """
    global _server, _server_thread, _started_port

    if _server_thread is not None and _server_thread.is_alive():
        return True, int(_started_port or port)

    try:
        server = ThreadingHTTPServer((host, port), BridgeHandler)
    except OSError:
        return False, port

    _server = server
    _started_port = port
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    _server_thread = th
    return True, port


def buffered_count() -> int:
    with _events_lock:
        return len(_events)


def listening_port() -> int | None:
    return _started_port


def clear_buffer() -> None:
    with _events_lock:
        _events.clear()


def fetch_events_http(base_url: str, limit: int) -> list[dict[str, Any]]:
    import urllib.error
    import urllib.request

    url = f"{base_url.rstrip('/')}/events?limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]
