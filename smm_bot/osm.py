"""Поиск лидов в OpenStreetMap через Overpass API.

Бесплатно, без API-ключей, данные открытые (ODbL). Главная ценность:
в OSM у объектов есть тег website — значит мы сразу видим бизнес БЕЗ сайта.

Overpass нестабилен и любит отдавать пустой ответ при перегрузке, поэтому:
  * делаем несколько попыток с паузами,
  * кешируем успешный ответ на диск,
  * отличаем "реально 0 объектов" от "сервер отдохнул" (пустой JSON при 200).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .models import Lead

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "osm_cache"
# Координаты городов кешируем отдельно и кладём в репозиторий: на хостинге
# Nominatim часто блокирует облачный IP, а этот кеш делает поиск независимым.
GEO_CACHE_DIR = Path(__file__).resolve().parent / "geo_cache"

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
PHOTON_URL = "https://photon.komoot.io/api/"
USER_AGENT = "smm-lead-finder/0.2 (+https://github.com/Chr0mat1x/smm-lead-bot)"

# Nominatim режет облачные IP (Render, VPS) и лимитирует 1 запрос/сек.
# Поэтому кешируем координаты городов на диск и держим запасной геокодер.

# Категории, которые вас интересуют: ключ — имя, значение — теги для Overpass.
CATEGORY_PRESETS: dict[str, list[tuple[str, str]]] = {
    "cafe": [("amenity", "cafe")],
    "restaurant": [("amenity", "restaurant")],
    "fast_food": [("amenity", "fast_food")],
    "bar": [("amenity", "bar"), ("amenity", "pub")],
    "bakery": [("shop", "bakery")],
    "banya": [("leisure", "sauna"), ("leisure", "spa")],
    "barber": [("shop", "hairdresser")],
    "beauty": [("shop", "beauty"), ("shop", "massage")],
    "gym": [("leisure", "fitness_centre"), ("leisure", "sports_centre")],
    "car_service": [("shop", "car_repair"), ("shop", "tyres")],
    "laundry": [("shop", "laundry"), ("shop", "dry_cleaning")],
    "florist": [("shop", "florist")],
    "pet": [("shop", "pet_grooming"), ("shop", "pet")],
    "auto_wash": [("amenity", "car_wash")],
    "diy": [("shop", "doityourself"), ("shop", "hardware")],
}

DEFAULT_CATEGORIES = ["cafe", "banya", "barber", "beauty", "bakery", "auto_wash", "car_service"]


@dataclass
class GeoArea:
    query: str
    bbox: tuple[float, float, float, float]  # south, west, north, east

    def as_overpass(self) -> str:
        s, w, n, e = self.bbox
        return f"{s},{w},{n},{e}"


def _geocode_nominatim(place: str, timeout: int) -> GeoArea | None:
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": place, "format": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    if resp.status_code == 429:  # превышен лимит — не ошибка места, а отказ сервера
        return None
    resp.raise_for_status()
    data = resp.json()
    if not data:
        return None
    s, n, w, e = (float(x) for x in data[0]["boundingbox"])
    return GeoArea(query=place, bbox=(s, w, n, e))


def _geocode_photon(place: str, timeout: int) -> GeoArea | None:
    """Запасной геокодер. Отдаёт точку + границы, поэтому сами расширяем
    точку до небольшой рамки — для поиска организаций этого достаточно."""
    resp = requests.get(
        PHOTON_URL,
        params={"q": place, "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
    )
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        return None
    props = features[0]
    extent = props.get("properties", {}).get("extent")
    if extent and len(extent) == 4:  # [west, north, east, south]
        w, n, e, s = (float(x) for x in extent)
        return GeoArea(query=place, bbox=(s, w, n, e))
    lon, lat = props["geometry"]["coordinates"]
    pad = 0.12  # примерно 13 км — город целиком
    return GeoArea(query=place, bbox=(lat - pad, lon - pad, lat + pad, lon + pad))


def geocode(place: str, timeout: int = 20, retries: int = 3, use_cache: bool = True) -> GeoArea:
    """Название города/района -> bounding box.

    Идём по цепочке: кеш -> Nominatim (с повторами) -> Photon. Так одна
    перегруженная служба не срывает поиск, что особенно важно на хостинге.
    """
    cache_file = GEO_CACHE_DIR / f"geo_{hashlib.sha256(place.strip().lower().encode()).hexdigest()[:16]}.json"
    if use_cache and cache_file.exists():
        s, w, n, e = json.loads(cache_file.read_text(encoding="utf-8"))
        return GeoArea(query=place, bbox=(s, w, n, e))

    last_error: Exception | None = None
    for attempt in range(retries):
        for lookup in (_geocode_nominatim, _geocode_photon):
            try:
                area = lookup(place, timeout)
            except requests.RequestException as exc:
                last_error = exc
                continue
            except (KeyError, ValueError, TypeError) as exc:
                last_error = exc
                continue
            if area is not None:
                if use_cache:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(list(area.bbox)), encoding="utf-8")
                return area
        time.sleep(1.5 * (attempt + 1))  # не долбим сервис при отказе
    raise RuntimeError(f"Не удалось определить координаты «{place}»: {last_error}")


def build_query(area: GeoArea, categories: list[str]) -> str:
    """Overpass QL: собираем запрос по всем выбранным категориям."""
    parts: list[str] = []
    for cat in categories:
        for tag_key, tag_value in CATEGORY_PRESETS.get(cat, []):
            parts.append(f'  nwr["{tag_key}"="{tag_value}"]({area.as_overpass()});')
    if not parts:
        raise ValueError("Пустой список категорий")
    return "[out:json][timeout:60];\n(\n" + "\n".join(parts) + "\n);\nout center tags;"


def _cache_path(query: str) -> Path:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def fetch(area: GeoArea, categories: list[str], retries: int = 5, timeout: int = 120,
          use_cache: bool = True) -> list[dict]:
    """Запрос к Overpass с повторами, перебором зеркал и кешем.

    Пустой ответ при HTTP 200 трактуем как сбой сервера и пробуем снова:
    для города это почти всегда признак перегрузки, а не отсутствие данных.
    """
    query = build_query(area, categories)
    cache_file = _cache_path(query)
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached:
            return cached

    last_error: Exception | None = None
    for attempt in range(retries):
        for endpoint in OVERPASS_ENDPOINTS:
            try:
                resp = requests.post(endpoint, data={"data": query},
                                     headers={"User-Agent": USER_AGENT}, timeout=timeout)
            except requests.RequestException as exc:
                last_error = exc
                continue
            if resp.status_code != 200:
                last_error = RuntimeError(f"{endpoint} -> HTTP {resp.status_code}")
                continue
            try:
                elements = resp.json().get("elements", [])
            except ValueError as exc:
                last_error = exc
                continue
            if not elements:
                last_error = RuntimeError(f"{endpoint} -> пустой ответ (вероятно перегрузка)")
                continue
            if use_cache:
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps(elements, ensure_ascii=False), encoding="utf-8")
            return elements
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Overpass недоступен после {retries} попыток: {last_error}")


def _first_tag(tags: dict, *keys: str) -> str:
    for key in keys:
        value = tags.get(key)
        if value:
            return value.strip()
    return ""


def parse_elements(elements: list[dict], city: str, source: str = "osm") -> list[Lead]:
    """Преобразование ответа Overpass в лиды. Только объекты без сайта.

    Объекты без имени пропускаем: без названия в OSM это почти всегда мусор,
    а персональное обращение без имени теряет смысл.
    """
    leads: list[Lead] = []
    for el in elements:
        tags = el.get("tags") or {}
        if not tags.get("name"):
            continue

        if _first_tag(tags, "website", "contact:website", "url"):
            continue  # сайт уже есть — не наш клиент

        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")

        address = " ".join(x for x in [
            _first_tag(tags, "addr:street"), _first_tag(tags, "addr:housenumber")
        ] if x)

        leads.append(Lead(
            source=source,
            source_id=f"{el.get('type')}/{el.get('id')}",
            name=tags["name"].strip(),
            category=_first_tag(tags, "amenity", "shop", "leisure"),
            city=city,
            address=address,
            lat=lat,
            lon=lon,
            has_website=False,
            phone=_first_tag(tags, "phone", "contact:phone", "contact:mobile"),
            email=_first_tag(tags, "email", "contact:email"),
            instagram=_first_tag(tags, "contact:instagram"),
            vk=_first_tag(tags, "contact:vk", "vk"),
            telegram=_first_tag(tags, "contact:telegram", "telegram"),
            notes=_first_tag(tags, "contact:facebook", "facebook"),
        ))
    return leads


def discover(place: str, categories: list[str] | None = None) -> list[Lead]:
    """Полный цикл: геокодинг -> запрос -> парсинг."""
    categories = categories or DEFAULT_CATEGORIES
    area = geocode(place)
    elements = fetch(area, categories)
    return parse_elements(elements, city=place)
