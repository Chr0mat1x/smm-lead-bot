"""Инструменты, которые LLM может вызывать сам.

Это ключевая часть: без инструментов ассистент только "разговаривает",
а с ними он реально правит базу. Вы пишете "это не тот контакт, он в другом
районе" — модель вызывает reject_lead, удаляет лид из выдачи и берёт следующий.

Каждый инструмент возвращает dict: {"ok": bool, "result": str, "data": dict}.
Возвращаемую строку модель видит и использует в ответе, поэтому пишем её
по-русски и по делу.
"""
from __future__ import annotations

from typing import Any

from .config import settings
from .message_generator import generate_message
from .models import Lead, LeadStatus
from . import osm
from .storage import Storage

# те же подписи, что в кнопках бота: в выдаче не должно быть английского "sauna"
CATEGORY_LABELS = {
    "cafe": "Кафе", "restaurant": "Ресторан", "fast_food": "Фастфуд", "bar": "Бар",
    "bakery": "Пекарня", "banya": "Баня/сауна", "barber": "Парикмахерская",
    "beauty": "Салон красоты", "gym": "Фитнес", "car_service": "Автосервис",
    "laundry": "Химчистка", "florist": "Цветочный", "pet": "Зооуслуги",
    "auto_wash": "Автомойка", "diy": "Строймагазин",
}


def category_label(value: str | None) -> str:
    """Подпись категории для выдачи.

    В базе категория — это значение тега OSM ("sauna", "hairdresser"),
    поэтому сначала смотрим прямые подписи, потом переводим через синонимы.
    """
    if not value:
        return "без категории"
    if value in CATEGORY_LABELS:
        return CATEGORY_LABELS[value]
    preset = osm.CATEGORY_ALIASES.get(value)

    return CATEGORY_LABELS.get(preset or "", value)


def contact_hint(lead: Lead) -> str:
    """Куда писать лиду и по какому каналу.

    Telegram и телефон — не равнозначные пути: в канал можно написать сразу,
    а по телефону владелец согласует разговор. Поэтому в короткой выдаче
    показываем оба, чтобы владелец выбирал сам.
    """
    tg = lead.tg
    if tg.kind == "channel" and tg.handle:
        return f"канал {tg.at or tg.url}"
    if lead.phone:
        return f"телефон {lead.phone}"
    return "контакт не указан"


# JSON-схемы для LLM (формат OpenAI tools, для Anthropic конвертируем в llm.py)
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_leads",
            "description": "Найти новые бизнесы без сайта в указанном городе. "
                           "Use when пользователь просит найти клиентов.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Город, например Тюмень"},
                    "categories": {
                        "type": "array",
                        # именно ключи пресетов: раньше здесь были значения тегов
                        # OSM (sauna, hairdresser), и поиск падал на них
                        "items": {"type": "string", "enum": sorted(osm.CATEGORY_PRESETS)},
                        "description": "Какие категории искать. Пусто = стандартный набор. "
                                       "Бери ключи из списка: banya — это бани и сауны.",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_leads",
            "description": "Показать из базы лиды с указанным статусом, по убыванию приоритета.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string",
                               "enum": [s.value for s in LeadStatus],
                               "description": "Статус лидов, по умолчанию new"},
                    "limit": {"type": "integer", "description": "Сколько показать, по умолчанию 5"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_lead",
            "description": "Получить полную карточку конкретного лида по его ключу.",
            "parameters": {
                "type": "object",
                "properties": {"lead_key": {"type": "string", "description": "Ключ лида, вида osm:node/123"}},
                "required": ["lead_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reject_lead",
            "description": "Пометить лид как неподходящий. Use when пользователь говорит, "
                           "что контакт не тот, не по адресу, не клиент, спам и т.п.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_key": {"type": "string"},
                    "reason": {"type": "string", "description": "Почему отклонили"},
                },
                "required": ["lead_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "do_not_contact",
            "description": "Добавить лид в чёрный список: больше никогда не писать.",
            "parameters": {
                "type": "object",
                "properties": {"lead_key": {"type": "string"}},
                "required": ["lead_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_message",
            "description": "Заменить текст письма для лида. Use when пользователь просит "
                           "переписать сообщение, сделать короче, убрать лишнее.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_key": {"type": "string"},
                    "text": {"type": "string", "description": "Новый текст сообщения"},
                },
                "required": ["lead_key", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "regenerate_message",
            "description": "Сгенерировать письмо для лида заново по шаблону, "
                           "с учётом пожеланий пользователя.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_key": {"type": "string"},
                    "hint": {"type": "string", "description": "Пожелание: тон, акцент, длина"},
                },
                "required": ["lead_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_contact",
            "description": "Исправить контакт лида: телефон, telegram, email, instagram, vk или адрес.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lead_key": {"type": "string"},
                    "field": {"type": "string",
                              "enum": ["phone", "telegram", "email", "instagram", "vk", "address"]},
                    "value": {"type": "string"},
                },
                "required": ["lead_key", "field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mark_contacted",
            "description": "Отметить, что пользователь уже отправил лиду сообщение.",
            "parameters": {
                "type": "object",
                "properties": {"lead_key": {"type": "string"}},
                "required": ["lead_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stats",
            "description": "Сколько лидов в базе и в каких статусах.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class ToolBox:
    """Выполняет инструменты поверх хранилища. Одна обёртка — одна база."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        # ключи лидов из последней выдачи — чтобы модель понимала «этот», «второй»
        self.last_shown: list[str] = []
        # город последнего поиска: поиск по Саратову не должен показывать Питер
        self.active_city: str = ""
        # категории последнего поиска: в выдаче «бани» не должно быть кафе
        self.active_categories: list[str] = []

    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return {"ok": False, "result": f"Неизвестный инструмент: {name}"}
        try:
            return handler(args)
        except Exception as exc:  # noqa: BLE001 — ошибку показываем модели, чтобы она исправилась
            return {"ok": False, "result": f"Ошибка инструмента {name}: {exc}"}

    # ---------- реализации ----------

    def _tool_search_leads(self, args: dict) -> dict:
        from .pipeline import run_discovery  # локальный импорт против цикла

        city = (args.get("city") or "").strip()
        if not city:
            return {"ok": False, "result": "Не указан город."}
        categories = args.get("categories") or None
        stats = run_discovery(self.storage, city, categories)
        self.active_city = stats.city or city
        self.active_categories = osm.category_tag_values(categories)
        self.storage.reset_shown_for_city(self.active_city)

        def pick(unseen: bool) -> list[Lead]:
            # лид годится, если ему вообще можно написать: канал или телефон.
            # Требовать только канал нельзя — в OSM у российских заведений
            # Telegram почти не указан, и выдача оставалась пустой.
            return self.storage.find_by_city(
                self.active_city, status=LeadStatus.NEW, unseen_only=unseen,
                categories=self.active_categories or None,
                contactable_only=True)[:10]

        leads = pick(unseen=True)
        if not leads:
            leads = pick(unseen=False)
        else:
            self.storage.mark_shown([l.key for l in leads])
        self.last_shown = [l.key for l in leads]
        city_label = leads[0].city if leads else self.active_city
        # Лиды кладём прямо в результат: иначе модель делает лишний круг
        # «tool -> show_leads -> tool -> ответ» и в сумме отвечает по минуте.
        if leads:
            # больше пяти не отдаём: модель честно переписывает весь список,
            # и ответ растёт до минуты
            head = leads[:5]
            lines = [f"{i+1}. {l.name} — {category_label(l.category)}, "
                     f"{l.city or 'адрес неизвестен'}"
                     f" | {contact_hint(l)} | скор {l.score} | ключ {l.key}"
                     for i, l in enumerate(head)]
            listing = "\n\nЛиды по приоритету:\n" + "\n".join(lines)
            if len(leads) > len(head):
                listing += f"\n...и ещё {len(leads) - len(head)}. Остальные — по кнопке «Следующие лиды»."
        else:
            total = len(self.storage.find_by_city(
                self.active_city, status=LeadStatus.NEW,
                categories=self.active_categories or None))
            listing = ("\n\nЛидов с контактом в этом городе не нашлось "
                       f"(всего бизнесов без сайта: {total}). Попробуйте другой город "
                       "или категорию.")
        return {
            "ok": True,
            "result": f"По городу {city_label}: {stats.as_text()}{listing}",
            "data": {"stats": stats.__dict__, "shown_keys": self.last_shown},
        }

    def _tool_show_leads(self, args: dict) -> dict:
        status_raw = args.get("status") or LeadStatus.NEW.value
        try:
            status = LeadStatus(status_raw)
        except ValueError:
            status = LeadStatus.NEW
        limit = int(args.get("limit") or 5)
        # Если недавно был поиск по городу — показываем из него, а не всю базу.
        # Категории тоже держим: после «найди бани» не должно быть кафе.
        # И только те, кому можно написать: канал или телефон.
        leads = self.storage.list_leads(status=status, limit=limit,
                                        city=self.active_city or None,
                                        categories=self.active_categories or None,
                                        contactable_only=True)
        self.last_shown = [l.key for l in leads]
        if not leads:
            where = f" по городу {self.active_city}" if self.active_city else ""
            return {"ok": True, "result": f"Лидов с контактом со статусом "
                                          f"{status.value}{where} нет."}
        lines = [f"{i+1}. {l.name} ({category_label(l.category)}, {l.city}) — "
                 f"{contact_hint(l)}, скор {l.score}, ключ {l.key}"
                 for i, l in enumerate(leads)]
        return {"ok": True, "result": "Найдены лиды:\n" + "\n".join(lines),
                "data": {"shown_keys": self.last_shown}}

    def _tool_get_lead(self, args: dict) -> dict:
        lead = self.storage.get(args.get("lead_key", ""))
        if not lead:
            return {"ok": False, "result": "Лид не найден."}
        return {"ok": True, "result": _describe(lead), "data": {"lead_key": lead.key}}

    def _tool_reject_lead(self, args: dict) -> dict:
        key = args.get("lead_key", "")
        if not self.storage.get(key):
            return {"ok": False, "result": "Лид не найден, отклонять нечего."}
        self.storage.set_status(key, LeadStatus.REJECTED)
        reason = args.get("reason") or "без причины"
        return {"ok": True, "result": f"Лид {key} отмечен как неподходящий ({reason}).",
                "data": {"lead_key": key, "status": LeadStatus.REJECTED.value}}

    def _tool_do_not_contact(self, args: dict) -> dict:
        key = args.get("lead_key", "")
        if not self.storage.get(key):
            return {"ok": False, "result": "Лид не найден."}
        self.storage.set_status(key, LeadStatus.DO_NOT_CONTACT)
        return {"ok": True, "result": f"Лид {key} в чёрном списке, больше не пишем.",
                "data": {"lead_key": key, "status": LeadStatus.DO_NOT_CONTACT.value}}

    def _tool_set_message(self, args: dict) -> dict:
        key = args.get("lead_key", "")
        text = (args.get("text") or "").strip()
        if not self.storage.get(key):
            return {"ok": False, "result": "Лид не найден."}
        if not text:
            return {"ok": False, "result": "Пустой текст, нечего сохранять."}
        self.storage.set_message(key, text)
        return {"ok": True, "result": f"Новый текст для {key} сохранён.",
                "data": {"lead_key": key, "message": text}}

    def _tool_regenerate_message(self, args: dict) -> dict:
        lead = self.storage.get(args.get("lead_key", ""))
        if not lead:
            return {"ok": False, "result": "Лид не найден."}
        text = generate_message(lead, your_name=settings.your_name)
        hint = (args.get("hint") or "").strip()
        if hint:
            text += f"\n\n(пожелание учтено: {hint})"
        self.storage.set_message(lead.key, text)
        return {"ok": True, "result": f"Письмо для {lead.key} перегенерировано.",
                "data": {"lead_key": lead.key, "message": text}}

    def _tool_update_contact(self, args: dict) -> dict:
        key = args.get("lead_key", "")
        field = args.get("field", "")
        value = (args.get("value") or "").strip()
        lead = self.storage.get(key)
        if not lead:
            return {"ok": False, "result": "Лид не найден."}
        allowed = {"phone", "telegram", "email", "instagram", "vk", "address"}
        if field not in allowed:
            return {"ok": False, "result": f"Поле {field} править нельзя. Доступно: {sorted(allowed)}"}
        with self.storage._conn() as conn:  # точечное обновление одного поля
            conn.execute(f"UPDATE leads SET {field} = ?, updated_at = datetime('now') WHERE key = ?",
                         (value, key))
        return {"ok": True, "result": f"У лида {key} поле {field} обновлено.",
                "data": {"lead_key": key, "field": field, "value": value}}

    def _tool_mark_contacted(self, args: dict) -> dict:
        key = args.get("lead_key", "")
        lead = self.storage.get(key)
        if not lead:
            return {"ok": False, "result": "Лид не найден."}
        self.storage.set_status(key, LeadStatus.CONTACTED)
        self.storage.log_send(key, lead.best_channel.value)
        return {"ok": True, "result": f"Отмечено: сообщение лиду {key} отправлено.",
                "data": {"lead_key": key, "status": LeadStatus.CONTACTED.value}}

    def _tool_stats(self, args: dict) -> dict:
        counts = self.storage.counts_by_status()
        lines = [f"всего: {self.storage.total()}",
                 f"отправлено сегодня: {self.storage.sent_today()}",
                 f"с публичным Telegram-каналом: {self.storage.count_tg_channels()}",
                 f"лидов с Telegram без проверки: {len(self.storage.unclassified_tg(10000))}"]
        lines += [f"{k}: {v}" for k, v in sorted(counts.items())]
        return {"ok": True, "result": "Статистика базы:\n" + "\n".join(lines)}


def _describe(lead: Lead) -> str:
    parts = [
        f"{lead.name} ({category_label(lead.category)}, {lead.city})",
        f"ключ: {lead.key}",
        f"статус: {lead.status.value}, скор: {lead.score}, канал: {lead.best_channel.value}",
    ]
    tg = lead.tg
    if tg.handle:
        parts.append(f"telegram: {tg.at or tg.url} ({tg.kind}), ссылка {tg.url}")
    for label, value in [("телефон", lead.phone), ("email", lead.email),
                         ("instagram", lead.instagram), ("vk", lead.vk),
                         ("адрес", lead.address)]:
        if value:
            parts.append(f"{label}: {value}")
    if lead.message:
        parts.append(f"письмо: {lead.message}")
    return " | ".join(parts)
