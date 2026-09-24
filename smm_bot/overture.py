"""Источник лидов из открытого датасета Overture Maps.

Зачем второй источник: в OpenStreetMap у российских заведений почти никогда
не указан Telegram, а у многих нет и телефона. Для Москвы и Санкт-Петербурга
по OSM набиралось ноль каналов, тогда как в базе уже лежали 1337 лидов
с телефоном. Overture — другой открытый датасет (данные из Facebook/Meta,
Microsoft и других), и он даёт заметно больше объектов с телефонами:
по Казани, например, 356 бизнесов без сайта с телефоном.

Данные лежат в открытом S3 в формате parquet, читаем их через DuckDB прямо
по HTTPS. Ключ не нужен, лимитов на запросы нет, платить не за что.
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

from .models import Lead

log = logging.getLogger("smm_bot.overture")

# Датасет обновляется ежемесячно. Актуальный список релизов можно посмотреть
# в S3-листинге, но за нас это делает LATEST: если версия уедет, поиск
# не упадёт, а просто сходит за списком (см. resolve_release).
S3_BUCKET = "overturemaps-us-west-2"
S3_REGION = "us-west-2"
FALLBACK_RELEASE = "2026-08-19.0"


@dataclass(frozen=True)
class Category:
    key: str                       # ключ пресета (как в osm.CATEGORY_PRESETS)
    label: str                     # человеческая подпись — и в OSM-тег, и в UI
    osm_value: str                 # значение тега OSM: в нём хранит категорию база
    overture_categories: tuple[str, ...]  # значения categories.primary в Overture


# Категории, ради которых вообще затевался бот: малый бизнес без сайта.
# osm_value совпадает со значениями тегов OSM, чтобы фильтры-по-категориям
# (`category IN (...)`) работали на объединённой базе, а не только на OSM.
CATEGORIES: tuple[Category, ...] = (
    Category("cafe", "Кафе", "cafe",
             ("cafe", "coffee_shop", "tea_room", "juice_bar", "ice_cream_shop",
              "dessert_shop", "bakery", "patisserie", "donut_shop")),
    Category("restaurant", "Ресторан", "restaurant",
             ("restaurant", "pizza_restaurant", "sushi_restaurant", "italian_restaurant",
              "japanese_restaurant", "chinese_restaurant", "steakhouse", "buffet_restaurant")),
    Category("fast_food", "Фастфуд", "fast_food",
             ("fast_food_restaurant", "food_truck", "hot_dog_stand")),
    Category("bar", "Бар", "bar",
             ("bar", "pub", "wine_bar", "cocktail_bar", "brewery")),
    Category("banya", "Баня/сауна", "sauna",
             ("sauna", "bathhouse", "public_bath")),
    Category("barber", "Парикмахерская", "hairdresser",
             ("barber", "hair_salon", "hair_extension_shop", "barber_shop")),
    Category("beauty", "Салон красоты", "beauty",
             ("beauty_salon", "nail_salon", "skin_care_clinic", "massage",
              "massage_clinic", "tanning_salon", "eyelash_service", "waxing_service",
              "spa", "spas", "health_spa")),
    Category("gym", "Фитнес", "fitness_centre",
             ("gym", "fitness_center", "yoga_studio", "pilates_studio",
              "martial_arts_gym", "boxing_gym", "dance_studio", "swimming_pool")),
    Category("car_service", "Автосервис", "car_repair",
             ("automotive_repair", "car_repair", "tire_shop", "auto_repair_shop",
              "car_inspection", "body_shop", "truck_repair")),
    Category("auto_wash", "Автомойка", "car_wash",
             ("car_wash", "car_detailing", "auto_detailing")),
    Category("laundry", "Химчистка", "laundry",
             ("laundry", "dry_cleaning", "laundromat", "laundry_service")),
    Category("florist", "Цветочный", "florist",
             ("florist", "flowers_and_gifts_shop", "flower_shop")),
    Category("pet", "Зооуслуги", "pet",
             ("pet_grooming", "pet_store", "pet_shop", "veterinarian", "pet_boarding")),
)

# ключ пресета -> категория: нужно, чтобы из списка пресетов бота собрать
# категории Overture
_BY_KEY: dict[str, Category] = {c.key: c for c in CATEGORIES}
# значение categories.primary в Overture -> категория
_BY_OVERTURE: dict[str, Category] = {
    name: c for c in CATEGORIES for name in c.overture_categories
}


def preset_keys() -> list[str]:
    return [c.key for c in CATEGORIES]


def supported_overture_categories(categories: list[str] | None) -> list[str]:
    """Какие значения categories.primary спросить у Overture.

    Принимаем и ключи пресетов бота, и значения тегов OSM («sauna»), потому
    что на вход категории приходят из разных мест.
    """
    if not categories:
        return [name for c in CATEGORIES for name in c.overture_categories]
    chosen: list[Category] = []
    for raw in categories:
        name = str(raw).strip().lower()
        cat = _BY_KEY.get(name) or _BY_OVERTURE.get(name)
        if cat and cat not in chosen:
            chosen.append(cat)
    if not chosen:
        return [name for c in CATEGORIES for name in c.overture_categories]
    return [name for c in chosen for name in c.overture_categories]


# Overture кладёт в одну категорию spa и бани, и салоны красоты, и массаж.
# По одному названию это не различить, поэтому смотрим на слова в названии.
_BATH_WORDS = ("баня", "бани", "сауна", "сауны", "bathhouse", "banya", "sauna")
_BEAUTY_WORDS = ("красот", "beauty", "nail", "ноготоч", "маникюр", "педикюр",
                 "ресниц", "бров", "brow", "lash", "hair", "barber", "stylist",
                 "массаж", "massage", "космет", "студия")


def _disambiguate(category: str, name: str, cat: "Category") -> "Category":
    """Развести бани и салоны внутри неоднозначной категории `spa`.

    Overture помечает одним значением `spa` и «Баня №10», и «Студия красоты».
    Раньше все они попадали в категорию «Баня/сауна», и барбершоп оказывался
    баней — и в подписи, и в сообщении, которое мы ему отправляем.
    """
    if category not in ("spa", "spas", "health_spa"):
        return cat
    low = (name or "").lower()
    if any(w in low for w in _BATH_WORDS):
        return _BY_KEY["banya"]
    if any(w in low for w in _BEAUTY_WORDS):
        return _BY_KEY["beauty"]
    return cat


def category_for(overture_category: str | None) -> Category | None:
    return _BY_OVERTURE.get((overture_category or "").strip().lower())


def build_query(bbox: tuple[float, float, float, float], categories: list[str] | None = None,
                release: str = FALLBACK_RELEASE, category_field: str = "categories.primary") -> str:
    """SQL для DuckDB: читаем parquet из S3 и оставляем нужный город.

    Качать весь датасет нельзя (десятки гигабайт), поэтому город отсекаем
    заранее. Ключевая деталь: отсекаем по обычному столбцу bbox, а не по
    geometry. Предикат по geometry заставляет читать все строки и занимает
    минуты, тогда как по bbox DuckDB пропускает файлы и группы строк по
    статистике — тот же город находится за секунду. ST_Intersects оставляем
    вторым условием: он дочищает пограничные объекты, но уже после отсечения.

    category_field — параметр, потому что в релизе 2026-09-23 Overture
    переименовала поле: categories.primary -> taxonomy.primary. Схема меняется
    без предупреждения, поэтому определяем её на месте (см. detect_category_field).
    """
    south, west, north, east = bbox
    names = ", ".join("'" + n.replace("'", "''") + "'" for n in supported_overture_categories(categories))
    path = f"s3://{S3_BUCKET}/release/{release}/theme=places/type=place/*.parquet"
    # пересечение рамок: объект попадает, даже если его bbox задевает город краем
    overlap = (f"places.bbox.xmax >= {west} AND places.bbox.xmin <= {east} "
               f"AND places.bbox.ymax >= {south} AND places.bbox.ymin <= {north}")
    return f"""
SELECT
    places.id,
    places.names.primary AS name,
    places.{category_field} AS category,
    places.confidence AS confidence,
    places.websites AS websites,
    places.phones AS phones,
    places.emails AS emails,
    places.socials AS socials,
    places.addresses[1].freeform AS address,
    places.addresses[1].locality AS locality,
    ST_Y(places.geometry) AS lat,
    ST_X(places.geometry) AS lon
FROM read_parquet('{path}') AS places
WHERE places.{category_field} IN ({names})
  AND {overlap}
""".strip()


def detect_category_field(con, release: str, timeout: int = 30) -> str:
    """Какое поле хранит категорию в этом релизе.

    В релизах до осени 2026 — categories.primary, в новых — taxonomy.primary.
    Тянуть датасет ради этого не нужно: DESCRIBE отдаёт схему из метаданных.
    """
    path = f"s3://{S3_BUCKET}/release/{release}/theme=places/type=place/*.parquet"
    try:
        columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()}
    except Exception as exc:  # noqa: BLE001 — схему не прочитали, пробуем привычное поле
        log.warning("Не удалось прочитать схему Overture %s: %s", release, exc)
        return "categories.primary"
    if "taxonomy" in columns:
        return "taxonomy.primary"
    return "categories.primary"


def _first_url(values) -> str:
    return str(values[0]).strip() if values else ""


def _normalize_phone(raw: str) -> str:
    return str(raw or "").strip()


def _social_handle(urls, needle: str) -> str:
    """Достаём ник из ссылки на соцсеть.

    В данных ссылки приходят как https://www.instagram.com/nik/, а в лид нам
    нужен nik — в таком виде он попадает в сообщение.
    """
    for url in urls or []:
        low = str(url).lower()
        if needle not in low:
            continue
        tail = str(url).split(needle, 1)[1] if needle in str(url) else ""
        handle = tail.strip("/").split("/")[0].split("?")[0]
        if handle:
            return handle
    return ""


def _is_telegram(url: str) -> bool:
    low = (url or "").lower()
    return "t.me/" in low or "telegram.me/" in low


def rows_to_leads(rows: list[tuple], city: str, source: str = "overture") -> list[Lead]:
    """Преобразование строк DuckDB в лиды. Только те, у кого нет сайта.

    Строкой сайта Overture считает любое значение в websites, поэтому ссылку
    на Telegram или соцсеть из websites переносим в нужный контакт, а сам лид
    оставляем «без сайта» — именно такие нам и нужны.
    """
    from .telegram_link import normalize, looks_like_telegram

    leads: list[Lead] = []
    for row in rows:
        (place_id, name, category, confidence, websites, phones, emails,
         socials, address, locality, lat, lon) = row
        if not name or not str(name).strip():
            continue
        cat = category_for(category)
        if cat is None:
            continue
        cat = _disambiguate(category, name, cat)

        websites = [str(w) for w in (websites or []) if w]
        # Telegram может лежать и в websites, и в socials
        telegram_candidates = [w for w in websites if looks_like_telegram(w) or _is_telegram(w)]
        telegram_candidates += [str(s) for s in (socials or []) if _is_telegram(str(s))]
        real_site = [w for w in websites if not (_is_telegram(w) or looks_like_telegram(w))]
        if real_site:
            continue  # сайт есть — не наш клиент

        ref = normalize(telegram_candidates[0] if telegram_candidates else "")

        leads.append(Lead(
            source=source,
            source_id=str(place_id),
            name=str(name).strip(),
            category=cat.osm_value,
            city=locality or city,
            address=str(address or "").strip(),
            lat=lat,
            lon=lon,
            has_website=False,
            phone=_normalize_phone(_first_url(phones)),
            email=_first_url(emails),
            instagram=_social_handle(socials, "instagram.com"),
            vk=_social_handle(socials, "vk.com"),
            telegram=ref.handle,
            tg_kind=ref.kind,
            notes=f"Overture, confidence {float(confidence):.2f}" if confidence else "",
        ))
    return leads


def is_available() -> bool:
    """Есть ли DuckDB. Без него источник просто не подключается."""
    try:
        import duckdb  # noqa: F401
    except ImportError:
        return False
    return True


_release_cache: dict[str, str] = {}


def resolve_release(timeout: int = 20) -> str:
    """Актуальная версия датасета. Датасет обновляется раз в месяц, и старая
    версия однажды исчезнет из S3 — тогда поиск молча вернул бы ноль объектов.
    Поэтому спрашиваем у S3 список релизов и берём последний.

    Список кешируется на время жизни процесса: бот работает сутками, а версия
    меняется раз в месяц.
    """
    if "latest" in _release_cache:
        return _release_cache["latest"]
    try:
        import requests

        resp = requests.get(
            f"https://{S3_BUCKET}.s3.amazonaws.com/",
            params={"list-type": "2", "prefix": "release/", "delimiter": "/", "max-keys": "1000"},
            timeout=timeout,
        )
        resp.raise_for_status()
        versions = re.findall(r"<Prefix>release/([^<]+)/</Prefix>", resp.text)
        versions = [v for v in versions if re.match(r"^\d{4}-\d{2}-\d{2}\.\d+$", v)]
        if versions:
            _release_cache["latest"] = sorted(versions)[-1]
            return _release_cache["latest"]
    except Exception as exc:  # noqa: BLE001 — не смогли узнать, работаем на известной
        log.warning("Не удалось определить версию Overture (%s), беру %s", exc, FALLBACK_RELEASE)
    _release_cache["latest"] = FALLBACK_RELEASE
    return FALLBACK_RELEASE


_conn = None
_conn_lock = threading.Lock()


def connect():
    """Одно соединение на процесс.

    Установка расширений httpfs/spatial при первом обращении стоит до пары
    минут, а у соединения есть свой кеш метаданных S3. Держим его открытым:
    иначе каждый поиск по городу заново тянул бы схему и тормозил.
    """
    global _conn
    if _conn is not None:
        return _conn
    with _conn_lock:
        if _conn is None:
            _conn = _connect()
    return _conn


def _connect():
    """Новое соединение с включённым чтением S3.

    httpfs нужен для parquet по HTTPS, spatial — для ST_Intersects и ST_Y/ST_X.
    """
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute(f"SET s3_region='{S3_REGION}';")
    con.execute("SET enable_progress_bar=false;")
    # в город может не быть сети нужной ширины: даём запас на медленный канал
    con.execute("SET http_timeout=120;")
    return con


def fetch(bbox: tuple[float, float, float, float], categories: list[str] | None = None,
          release: str | None = None, limit: int = 0) -> list[tuple]:
    """Строки Overture по рамке города. Исключения наружу пробрасываем: их
    ловит вызывающий и продолжает работать на одном OSM (см. pipeline).
    """
    if not is_available():
        raise RuntimeError("DuckDB не установлен — источник Overture недоступен.")

    release = release or resolve_release()
    con = connect()
    field = detect_category_field(con, release)
    query = build_query(bbox, categories, release=release, category_field=field)
    if limit:
        query += f"\nLIMIT {int(limit)}"
    return con.execute(query).fetchall()


def fetch_leads(bbox: tuple[float, float, float, float], city: str,
                categories: list[str] | None = None,
                release: str | None = None) -> list[Lead]:
    rows = fetch(bbox, categories, release=release)
    return rows_to_leads(rows, city)


