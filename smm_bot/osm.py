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
# Ищем только по России: иначе «Казани» находится как село Кажани в Македонии.
NOMINATIM_COUNTRY = "ru"
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

# Словарь синонимов: и значения тегов OSM, и русские слова владельца.
# Без него «найди бани» ломалось: нейросеть видела в схеме инструмента
# значение тега "sauna", а поиск знает только ключ пресета "banya".
CATEGORY_ALIASES: dict[str, str] = {
    # значения тегов OSM -> ключ пресета
    "sauna": "banya", "spa": "banya",
    "hairdresser": "barber",
    "massage": "beauty",
    "fitness_centre": "gym", "sports_centre": "gym", "gym": "gym",
    "car_repair": "car_service", "tyres": "car_service",
    "dry_cleaning": "laundry",
    "car_wash": "auto_wash",
    "doityourself": "diy", "hardware": "diy",
    "pet_grooming": "pet",
    "pub": "bar",
    # русские слова
    "баня": "banya", "бани": "banya", "банный": "banya", "сауна": "banya", "спа": "banya",
    "кафе": "cafe", "кофейня": "cafe", "кофе": "cafe", "кофейни": "cafe",
    "барбершоп": "barber", "парикмахерская": "barber", "стрижка": "barber",
    "салон": "beauty", "красота": "beauty", "массаж": "beauty",
    "автомойка": "auto_wash", "мойка": "auto_wash",
    "автосервис": "car_service", "шиномонтаж": "car_service", "сто": "car_service",
    "пекарня": "bakery", "хлеб": "bakery",
    "прачечная": "laundry", "химчистка": "laundry",
    "цветы": "florist", "цветочный": "florist",
    "зоомагазин": "pet", "груминг": "pet",
    "фитнес": "gym", "спортзал": "gym",
    "ресторан": "restaurant",
    "бар": "bar", "паб": "bar",
    "стройматериалы": "diy",
}


def normalize_categories(categories: list[str] | str | None) -> list[str]:
    """Приводим категории к ключам CATEGORY_PRESETS, отбрасывая мусор."""
    if categories is None:
        return []
    if isinstance(categories, str):
        categories = categories.split(",")
    result: list[str] = []
    for raw in categories:
        name = str(raw).strip().lower()
        if not name:
            continue
        key = name if name in CATEGORY_PRESETS else CATEGORY_ALIASES.get(name)
        if key and key not in result:
            result.append(key)
    return result


def category_tag_values(categories: list[str] | str | None) -> list[str]:
    """Значения тегов OSM для категорий — их и хранит база.

    «Баня» в базе лежит как «sauna» или «spa», поэтому фильтровать по ключу
    пресета нельзя: вернём и сами теги, и ключ на случай старых записей.
    """
    values: list[str] = []
    for cat in normalize_categories(categories):
        if cat not in values:
            values.append(cat)
        for _, tag_value in CATEGORY_PRESETS.get(cat, []):
            if tag_value not in values:
                values.append(tag_value)
    return values


# `leisure=spa` в OSM, как и `spa` в Overture, — это и бани, и салоны красоты,
# и массаж. По одному тегу их не различить, поэтому смотрим на слова в названии.
# Так «Студия красоты» не попадает в категорию «Баня/сауна» и не получает
# рекламу бани. Логика общая для обоих источников (см. overture.py).
_BATH_WORDS = ("баня", "бани", "банный", "сауна", "сауны", "bathhouse", "banya", "sauna")
_BEAUTY_WORDS = ("красот", "beauty", "nail", "ноготоч", "маникюр", "педикюр",
                 "ресниц", "бров", "brow", "lash", "hair", "barber", "stylist",
                 "массаж", "massage", "космет", "студия")


def resolve_spa_category(name: str) -> str:
    """Развести бани и салоны внутри неоднозначной категории `spa`.

    Возвращает значение тега OSM: `sauna` для бань, `spa` для салонов.
    Без подсказки в названии считаем салоном — салонов в этой группе больше.
    """
    low = (name or "").lower()
    if any(w in low for w in _BATH_WORDS):
        return "sauna"
    return "spa"


@dataclass
class GeoArea:
    query: str
    bbox: tuple[float, float, float, float]  # south, west, north, east
    name: str = ""  # каноническое название; в карточках не показываем «Тюмени»

    def as_overpass(self) -> str:
        s, w, n, e = self.bbox
        return f"{s},{w},{n},{e}"


def bbox_is_usable(bbox: tuple[float, float, float, float]) -> bool:
    """Отсекаем «точку» вместо города.

    Nominatim на падежную форму («Тюмени») возвращает рамку площадью ~0 —
    поиск по ней даёт 2-3 объекта, и это выглядит как «бот ничего не нашёл».
    Порог грубый: он отделяет точку от настоящего города, а не районы.
    """
    s, w, n, e = bbox
    return (n - s) * (e - w) > 0.002


# Окончания, которые появляются, когда город называют в падеже.
_CASE_ENDINGS = ("и", "е", "у", "а", "ой", "ем", "ом", "ы")


def name_variants(place: str) -> list[str]:
    """Варианты написания города: «Тюмени» -> «Тюмен» -> «Тюмень».

    Нейросеть почти всегда ставит город в падеж («найди кафе в Тюмени»),
    а геокодер такую форму понимает плохо. Поэтому пробуем основу слова.
    """
    base = place.strip()
    variants = [base]
    low = base.lower()
    for end in _CASE_ENDINGS:
        if low.endswith(end) and len(base) - len(end) >= 3:
            stem = base[: -len(end)]
            if stem not in variants:
                variants.append(stem)
    # «Тюмен» -> «Тюмень»: восстанавливаем мягкий знак для женских названий
    if base and not low.endswith("ь"):
        stem = variants[1] if len(variants) > 1 else base
        if stem and stem[-1] in "нлтсрмкзв":
            soft = stem + "ь"
            if soft not in variants:
                variants.append(soft)
    return variants


def _geocode_nominatim(place: str, timeout: int) -> GeoArea | None:
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": place, "format": "json", "limit": 1, "addressdetails": 1,
                "countrycodes": NOMINATIM_COUNTRY},
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
    address = data[0].get("address") or {}
    nice = (address.get("city") or address.get("town") or address.get("village")
            or address.get("municipality") or data[0].get("name") or place)
    return GeoArea(query=place, bbox=(s, w, n, e), name=nice)


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
    # Photon ищет по всему миру: «Казани» он находит как село в Македонии.
    country = (props.get("properties") or {}).get("countrycode")
    if country and country != NOMINATIM_COUNTRY.upper():
        return None
    extent = props.get("properties", {}).get("extent")
    props_map = props.get("properties") or {}
    nice = props_map.get("city") or props_map.get("name") or place
    if extent and len(extent) == 4:  # [west, north, east, south]
        w, n, e, s = (float(x) for x in extent)
        return GeoArea(query=place, bbox=(s, w, n, e), name=nice)
    lon, lat = props["geometry"]["coordinates"]
    pad = 0.12  # примерно 13 км — город целиком
    return GeoArea(query=place, bbox=(lat - pad, lon - pad, lat + pad, lon + pad), name=nice)


def geocode(place: str, timeout: int = 20, retries: int = 3, use_cache: bool = True) -> GeoArea:
    """Название города/района -> bounding box.

    Идём по цепочке: кеш -> Nominatim (с повторами) -> Photon. Так одна
    перегруженная служба не срывает поиск, что особенно важно на хостинге.
    """
    def _cache_file(name: str) -> Path:
        digest = hashlib.sha256(name.strip().lower().encode()).hexdigest()[:16]
        return GEO_CACHE_DIR / f"geo_{digest}.json"

    # Кеш лежит под начальной формой названия («тюмень»), а нейросеть присылает
    # падеж («тюмени») — поэтому смотрим кеш и по вариантам написания.
    if use_cache:
        for name in name_variants(place):
            candidate = _cache_file(name)
            if not candidate.exists():
                continue
            cached = json.loads(candidate.read_text(encoding="utf-8"))
            # старые записи — просто список координат, новые — объект с названием
            bbox = tuple(cached["bbox"]) if isinstance(cached, dict) else tuple(cached)
            if bbox_is_usable(bbox):
                name = cached.get("name", "") if isinstance(cached, dict) else ""
                return GeoArea(query=place, bbox=bbox, name=name or place)

    cache_file = _cache_file(place)

    last_error: Exception | None = None
    candidates = name_variants(place)
    for attempt in range(retries):
        for name in candidates:
            for lookup in (_geocode_nominatim, _geocode_photon):
                try:
                    area = lookup(name, timeout)
                except requests.RequestException as exc:
                    last_error = exc
                    continue
                except (KeyError, ValueError, TypeError) as exc:
                    last_error = exc
                    continue
                if area is None:
                    continue
                if not bbox_is_usable(area.bbox):
                    # точка вместо города — пробуем другое написание
                    last_error = RuntimeError(f"«{name}» -> рамка нулевой площади")
                    continue
                if use_cache:
                    GEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    payload = {"bbox": list(area.bbox), "name": area.name}
                    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return area
        time.sleep(1.5 * (attempt + 1))  # не долбим сервис при отказе
    raise RuntimeError(f"Не удалось определить координаты «{place}»: {last_error}")


def build_query(area: GeoArea, categories: list[str] | str) -> str:
    """Overpass QL: собираем запрос по всем выбранным категориям.

    Категории нормализуем: нейросеть в вызове инструмента может передать как
    список, так и одну строку ("cafe"), а иногда строку через запятую. Ещё она
    любит значения тегов OSM ("sauna") вместо ключей пресетов ("banya") —
    поэтому неизвестное имя не роняем, а переводим через синонимы.
    """
    cats = normalize_categories(categories)
    parts: list[str] = []
    for cat in cats:
        for tag_key, tag_value in CATEGORY_PRESETS.get(cat, []):
            parts.append(f'  nwr["{tag_key}"="{tag_value}"]({area.as_overpass()});')
    if not parts:
        raise ValueError("Не понял категории. Доступные: "
                         + ", ".join(sorted(CATEGORY_PRESETS)))
    return "[out:json][timeout:60];\n(\n" + "\n".join(parts) + "\n);\nout center tags;"


def _cache_path(query: str) -> Path:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{digest}.json"


def _is_complete(data: dict) -> bool:
    """Overpass при таймауте отдаёт ЧАСТЬ данных и поле remark.

    Такой ответ нельзя ни возвращать, ни кешировать: иначе город навсегда
    «находится» из трёх объектов (проверено на живом сервисе).
    """
    if data.get("remark"):
        return False
    return True


def fetch(area: GeoArea, categories: list[str] | str, retries: int = 5, timeout: int = 120,
          use_cache: bool = True) -> list[dict]:
    """Запрос к Overpass с повторами, перебором зеркал и кешем.

    Пустой ответ при HTTP 200 трактуем как сбой сервера и пробуем снова:
    для города это почти всегда признак перегрузки, а не отсутствие данных.
    """
    query = build_query(area, categories)
    cache_file = _cache_path(query)
    if use_cache and cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
        except ValueError:
            cached = None
        # кеш старого формата мог сохранить неполный ответ — тогда выбрасываем
        if isinstance(cached, list) and cached and len(cached) > 3:
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
                data = resp.json()
            except ValueError as exc:
                last_error = exc
                continue
            elements = data.get("elements", [])
            if not elements:
                last_error = RuntimeError(f"{endpoint} -> пустой ответ (вероятно перегрузка)")
                continue
            if not _is_complete(data):
                last_error = RuntimeError(f"{endpoint} -> неполный ответ (таймаут на сервере)")
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

    Отдельный случай — Telegram в поле website. Бизнес без сайта часто пишет
    там ссылку на свой канал, и раньше такие объекты выбрасывались как «уже
    есть сайт», хотя сайта у них нет. Такой объект считаем лидом без сайта,
    а ссылку переносим в telegram.
    """
    from .telegram_link import looks_like_telegram, normalize

    leads: list[Lead] = []
    for el in elements:
        tags = el.get("tags") or {}
        if not tags.get("name"):
            continue

        website = _first_tag(tags, "website", "contact:website", "url")
        telegram = _first_tag(tags, "contact:telegram", "telegram")
        if website and looks_like_telegram(website):
            telegram = telegram or website
            website = ""  # это не сайт, а канал — объект остаётся «без сайта»
        if website:
            continue  # сайт уже есть — не наш клиент

        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")

        address = " ".join(x for x in [
            _first_tag(tags, "addr:street"), _first_tag(tags, "addr:housenumber")
        ] if x)

        ref = normalize(telegram)
        category = _first_tag(tags, "amenity", "shop", "leisure")
        if category == "spa":
            category = resolve_spa_category(tags["name"])
        leads.append(Lead(
            source=source,
            source_id=f"{el.get('type')}/{el.get('id')}",
            name=tags["name"].strip(),
            category=category,
            city=city,
            address=address,
            lat=lat,
            lon=lon,
            has_website=False,
            phone=_first_tag(tags, "phone", "contact:phone", "contact:mobile"),
            email=_first_tag(tags, "email", "contact:email"),
            instagram=_first_tag(tags, "contact:instagram"),
            vk=_first_tag(tags, "contact:vk", "vk"),
            telegram=ref.handle,
            tg_kind=ref.kind,
            notes=_first_tag(tags, "contact:facebook", "facebook"),
        ))
    return leads


def discover(place: str, categories: list[str] | None = None) -> list[Lead]:
    """Полный цикл: геокодинг -> запрос -> парсинг."""
    categories = categories or DEFAULT_CATEGORIES
    area = geocode(place)
    elements = fetch(area, categories)
    return parse_elements(elements, city=area.name or place)
