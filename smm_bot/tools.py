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
from .scoring import CATEGORY_WEIGHT
from .storage import Storage

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
                        "items": {"type": "string", "enum": sorted(CATEGORY_WEIGHT.keys())},
                        "description": "Какие категории искать. Пусто = стандартный набор.",
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
        self.storage.reset_shown_for_city(self.active_city)

        def pick(unseen: bool) -> list[Lead]:
            return [l for l in self.storage.find_by_city(self.active_city, status=LeadStatus.NEW,
                                                         unseen_only=unseen)
                    if l.reachable][:10]

        leads = pick(unseen=True)
        if not leads:
            leads = pick(unseen=False)
        else:
            self.storage.mark_shown([l.key for l in leads])
        self.last_shown = [l.key for l in leads]
        city_label = leads[0].city if leads else self.active_city
        return {
            "ok": True,
            "result": f"По городу {city_label}: {stats.as_text()}",
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
        leads = [l for l in self.storage.list_leads(status=status, limit=limit,
                                                    city=self.active_city or None)
                 if l.reachable]
        self.last_shown = [l.key for l in leads]
        if not leads:
            where = f" по городу {self.active_city}" if self.active_city else ""
            return {"ok": True, "result": f"Лидов со статусом {status.value}{where} нет."}
        lines = [f"{i+1}. {l.name} ({l.category}, {l.city}) — скор {l.score}, ключ {l.key}"
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
        lines = [f"всего: {self.storage.total()}", f"отправлено сегодня: {self.storage.sent_today()}"]
        lines += [f"{k}: {v}" for k, v in sorted(counts.items())]
        return {"ok": True, "result": "Статистика базы:\n" + "\n".join(lines)}


def _describe(lead: Lead) -> str:
    parts = [
        f"{lead.name} ({lead.category}, {lead.city})",
        f"ключ: {lead.key}",
        f"статус: {lead.status.value}, скор: {lead.score}, канал: {lead.best_channel.value}",
    ]
    for label, value in [("телефон", lead.phone), ("telegram", lead.telegram),
                         ("email", lead.email), ("instagram", lead.instagram),
                         ("vk", lead.vk), ("адрес", lead.address)]:
        if value:
            parts.append(f"{label}: {value}")
    if lead.message:
        parts.append(f"письмо: {lead.message}")
    return " | ".join(parts)
