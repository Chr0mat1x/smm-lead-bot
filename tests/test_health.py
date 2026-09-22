"""Тесты health-сервера.

Сервер реально поднимается в потоке и опрашивается по HTTP — моков нет.
Проверяем в том числе, что ошибка базы не роняет сервер, а отдаётся как 500.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smm_bot.health import start_health_server
from smm_bot.models import Lead
from smm_bot.storage import Storage


@pytest.fixture()
def running_server(storage):
    server = start_health_server(storage, 0)  # 0 = свободный порт
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}", storage
    server.shutdown()


@pytest.fixture()
def storage(tmp_path) -> Storage:
    return Storage(tmp_path / "health.sqlite3")


def test_health_returns_ok_and_counts(running_server) -> None:
    base, store = running_server
    store.upsert_lead(Lead(source="osm", source_id="node/1", name="Баня", category="sauna",
                           city="Тюмень", score=10))
    with urllib.request.urlopen(f"{base}/health", timeout=5) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read())

    assert payload["status"] == "ok"
    assert payload["leads"] == 1
    assert payload["sent_today"] == 0


def test_root_returns_plain_text(running_server) -> None:
    base, _ = running_server
    with urllib.request.urlopen(base, timeout=5) as resp:
        assert resp.status == 200
        assert b"smm_bot is running" in resp.read()


def test_health_reports_error_when_storage_broken(running_server) -> None:
    """Если база недоступна, отдаём 500, а не падаем всем процессом."""
    base, store = running_server
    store.total = lambda: (_ for _ in ()).throw(RuntimeError("база отвалилась"))

    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{base}/health", timeout=5)
    assert exc.value.code == 500
    payload = json.loads(exc.value.read())
    assert payload["status"] == "error"
    assert "база отвалилась" in payload["detail"]
