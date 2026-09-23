"""Небольшой HTTP-сервер здоровья.

Нужен там, где платформа ждёт открытый порт (Render, Railway, Fly и т.п.):
если порт не слушается, платформа считает сервис упавшим и перезапускает его.

Запускается только если задан PORT. Локально можно не поднимать.

    PORT=8080 python -m smm_bot.bot
    curl localhost:8080/health  ->  {"status": "ok", ...}
"""
from __future__ import annotations

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

log = logging.getLogger("smm_bot.health")


def _version() -> str:
    """Короткий хеш коммита, который сейчас задеплоен.

    Render отдаёт его в RENDER_GIT_COMMIT — по нему видно, подхватил ли сервис
    последний пуш, не открывая панель хостинга.
    """
    return (os.getenv("RENDER_GIT_COMMIT") or os.getenv("GIT_COMMIT") or "dev")[:7]


def _make_handler(storage, stats=None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass  # не засоряем логи запросами мониторинга

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/health"):
                try:
                    payload = {
                        "status": "ok",
                        "version": _version(),
                        "leads": storage.total(),
                        "sent_today": storage.sent_today(),
                    }
                    if stats:
                        payload.update(stats())
                    body = json.dumps(payload, ensure_ascii=False).encode()
                    code = 200
                except Exception as exc:  # noqa: BLE001 — база недоступна, честно говорим об этом
                    body = json.dumps({"status": "error", "detail": str(exc)}).encode()
                    code = 500
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"smm_bot is running")

    return Handler


def start_health_server(storage, port: int, stats=None) -> HTTPServer:
    """Поднимает сервер в фоновом потоке и возвращает его."""
    server = HTTPServer(("0.0.0.0", port), _make_handler(storage, stats))
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="health")
    thread.start()
    log.info("Health-сервер слушает порт %s (GET /health)", port)
    return server
