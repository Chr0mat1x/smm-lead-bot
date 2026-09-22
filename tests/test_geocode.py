"""Геокодинг: кеш и запасной сервис.

На хостинге Nominatim часто блокирует облачный IP, поэтому проверяем именно
устойчивость: поиск должен пережить недоступность обоих геокодеров за счёт
кеша координат, который лежит в репозитории.
"""
from __future__ import annotations

import json

import pytest

from smm_bot import osm


def test_geo_cache_is_bundled():
    """Кеш координат крупных городов должен ехать вместе с кодом."""
    files = list(osm.GEO_CACHE_DIR.glob("geo_*.json"))
    assert len(files) >= 50, f"в geo_cache только {len(files)} городов"


def test_geocode_reads_from_cache_without_network():
    """Даже если оба сервиса недоступны, известный город определяется."""
    def boom(*_a, **_k):
        raise RuntimeError("сеть недоступна")

    original = (osm._geocode_nominatim, osm._geocode_photon)
    osm._geocode_nominatim, osm._geocode_photon = boom, boom
    try:
        for city in ("Тюмень", "Казань", "Москва"):
            area = osm.geocode(city)
            s, w, n, e = area.bbox
            assert s < n and w < e, f"{city}: некорректный bbox {area.bbox}"
    finally:
        osm._geocode_nominatim, osm._geocode_photon = original


def test_degenerate_cached_bbox_is_rejected(tmp_path, monkeypatch):
    """Старые записи с рамкой-точкой («Тюмени») не должны приниматься."""
    monkeypatch.setattr(osm, "GEO_CACHE_DIR", tmp_path)
    import hashlib

    key = hashlib.sha256("тюмени".encode()).hexdigest()[:16]
    (tmp_path / f"geo_{key}.json").write_text(json.dumps([57.14, 65.56, 57.15, 65.56]),
                                              encoding="utf-8")
    original = (osm._geocode_nominatim, osm._geocode_photon)
    osm._geocode_nominatim = lambda q, t: osm.GeoArea(q, (57.0, 65.2, 57.3, 65.8), "Тюмень")
    osm._geocode_photon = lambda q, t: None
    try:
        area = osm.geocode("Тюмени", retries=1)
        assert area.bbox == (57.0, 65.2, 57.3, 65.8), "точка из кеша не должна приниматься"
        assert area.name == "Тюмень"
    finally:
        osm._geocode_nominatim, osm._geocode_photon = original


def test_case_forms_are_tried():
    """«найди в Тюмени» — нейросеть ставит падеж, геокодер его не понимает."""
    assert osm.name_variants("Тюмени")[:2] == ["Тюмени", "Тюмен"]
    assert "Тюмень" in osm.name_variants("Тюмени")
    assert osm.name_variants("Ярославле")[-1] == "Ярославль"


def test_case_form_found_in_cache_offline():
    """«Тюмени» должно находиться из кеша, где лежит «Тюмень».

    На хостинге геокодеры недоступны, а бот получает именно падеж.
    """
    def boom(*_a, **_k):
        raise RuntimeError("сеть недоступна")

    original = (osm._geocode_nominatim, osm._geocode_photon)
    osm._geocode_nominatim, osm._geocode_photon = boom, boom
    try:
        for form, expected in [("Тюмени", "Тюмень"), ("Казани", "Казань"),
                               ("Ярославле", "Ярославль")]:
            area = osm.geocode(form)
            assert area.name == expected, f"{form}: получено {area.name!r}"
            assert osm.bbox_is_usable(area.bbox), f"{form}: рамка нулевой площади"
    finally:
        osm._geocode_nominatim, osm._geocode_photon = original


def test_geocode_unknown_place_raises_clearly():
    """Неизвестное место — понятная ошибка, а не молчаливый ноль лидов."""
    original = (osm._geocode_nominatim, osm._geocode_photon)
    osm._geocode_nominatim = lambda *a, **k: None
    osm._geocode_photon = lambda *a, **k: None
    try:
        with pytest.raises(RuntimeError, match="координаты"):
            osm.geocode("Абракадабрия-которой-нет", retries=1, use_cache=False)
    finally:
        osm._geocode_nominatim, osm._geocode_photon = original


def test_photon_fallback_builds_area():
    """Отказ Nominatim не должен срывать поиск — подхватывает Photon."""
    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"features": [{
                "geometry": {"coordinates": [65.5, 57.1]},
                "properties": {"extent": [65.2, 57.3, 65.9, 57.0]},
            }]}

    calls: list[str] = []
    original_get = osm.requests.get

    def fake_get(url, *a, **k):
        calls.append(url)
        if "nominatim" in url:
            raise osm.requests.RequestException("429 Too Many Requests")
        return FakeResponse()

    osm.requests.get = fake_get
    try:
        area = osm.geocode("Выдуманск", retries=1, use_cache=False)
        assert any("nominatim" in c for c in calls), "Nominatim не был испробован"
        assert any("photon" in c for c in calls), "Photon не был использован"
        assert area.bbox == (57.0, 65.2, 57.3, 65.9)
    finally:
        osm.requests.get = original_get


def test_cache_file_written(tmp_path, monkeypatch):
    """Успешный геокодинг кладёт координаты в кеш."""
    monkeypatch.setattr(osm, "GEO_CACHE_DIR", tmp_path)
    original = (osm._geocode_nominatim, osm._geocode_photon)
    osm._geocode_nominatim = lambda *a, **k: osm.GeoArea("X", (1.0, 2.0, 3.0, 4.0))
    osm._geocode_photon = lambda *a, **k: None
    try:
        area = osm.geocode("НовыйГород", use_cache=True)
        assert area.bbox == (1.0, 2.0, 3.0, 4.0)
        saved = list(tmp_path.glob("geo_*.json"))
        assert len(saved) == 1
        payload = json.loads(saved[0].read_text(encoding="utf-8"))
        assert payload["bbox"] == [1.0, 2.0, 3.0, 4.0]
    finally:
        osm._geocode_nominatim, osm._geocode_photon = original
