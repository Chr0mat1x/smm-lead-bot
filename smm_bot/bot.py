"""Telegram-бот для управления лидами.

Запуск:  python -m smm_bot.bot

Бот не рассылает ничего сам. Он показывает лиды, готовые сообщения и
кнопки "одобрить / отклонить / отправил / не писать". Отправку вы
подтверждаете вручную — так вы не теряете аккаунт и не спамите наугад.
"""
from __future__ import annotations

import asyncio
import html
import logging
import os
import re
import time

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (CallbackQuery, ErrorEvent, FSInputFile, InlineKeyboardButton,
                           InlineKeyboardMarkup, KeyboardButton, Message,
                           ReplyKeyboardMarkup)

from .agent import Agent
from .config import settings
from .health import start_health_server
from .llm import build_llm
from .models import Channel, Lead, LeadStatus
from .osm import category_tag_values
from .pipeline import export_leads_csv, prepare_messages, recheck_telegram, run_discovery
from .scoring import CATEGORY_WEIGHT
from .storage import Storage
from .tools import category_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smm_bot")

# сколько ждём ответ нейросети, прежде чем сказать «не дождался»
AGENT_TIMEOUT = 120

storage = Storage(settings.db_path)
dp = Dispatcher()

# Состояние процесса наружу — через /health. На Render нет доступа к логам,
# поэтому «бот жив, но не отвечает» должен быть виден снаружи.
runtime_stats = {"handled": 0, "errors": 0, "last_update": "", "last_error": "",
                 "started_at": time.time()}


def stats_payload() -> dict:
    """Показатели процесса для /health.

    uptime отличает стабильный процесс от перезапускающегося: если он сбрасывается
    при каждом опросе, хостинг убивает бота, и «не отвечает» — про это.
    """
    return {**runtime_stats, "uptime_sec": int(time.time() - runtime_stats["started_at"])}


@dp.update.outer_middleware()
async def count_updates(handler, event, data):
    """Считаем дошедшие обновления: видно, получает ли бот сообщения вообще."""
    runtime_stats["handled"] += 1
    runtime_stats["last_update"] = type(event.event).__name__
    return await handler(event, data)



CATEGORY_LABELS = {
    "cafe": "Кафе", "restaurant": "Рестораны", "fast_food": "Фастфуд", "bar": "Бары",
    "bakery": "Пекарни", "banya": "Бани и сауны", "barber": "Парикмахерские",
    "beauty": "Салоны красоты", "gym": "Фитнес", "car_service": "Автосервисы",
    "laundry": "Химчистки", "florist": "Цветочные", "pet": "Зооуслуги",
    "auto_wash": "Автомойки", "diy": "Строймагазины",
}
DEFAULT_CATEGORIES_UI = ["cafe", "banya", "barber", "beauty", "bakery", "auto_wash", "car_service"]

# Выбранные пользователем категории (по chat_id). В памяти — этого достаточно для одного владельца.
selected_categories: dict[int, list[str]] = {}


def is_allowed(message_or_call) -> bool:
    if not settings.allowed_user_ids:
        return True  # список не задан — бот открыт только тому, у кого есть токен
    user = getattr(message_or_call, "from_user", None)
    return bool(user and user.id in settings.allowed_user_ids)


async def guard(message) -> bool:
    """Проверка доступа, которая объясняет отказ.

    Молчаливый `return` уже стоил часа диагностики: «бот не отвечает», а
    причины не видно. Поэтому показываем собеседнику его собственный ID —
    это единственное, что нужно, чтобы починить ALLOWED_USER_IDS.
    """
    if is_allowed(message):
        return True
    user = getattr(message, "from_user", None)
    uid = user.id if user else "неизвестен"
    log.warning("Отклонил доступ: id=%s username=@%s (ALLOWED_USER_IDS=%s)",
                uid, getattr(user, "username", None), settings.allowed_user_ids)
    await message.answer(
        f"Бот приватный.\n\nВаш Telegram ID: <code>{uid}</code>\n"
        "Добавьте его в переменную ALLOWED_USER_IDS на хостинге (через запятую, "
        "если не один) и напишите снова."
    )
    return False


def main_menu() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🔍 Найти клиентов")],
            [KeyboardButton(text="📋 Следующие лиды"), KeyboardButton(text="📊 Статистика")],
            [KeyboardButton(text="🗂 Категории"), KeyboardButton(text="📤 Экспорт CSV")],
        ],
        resize_keyboard=True,
    )


def categories_menu(chat_id: int) -> InlineKeyboardMarkup:
    chosen = set(selected_categories.get(chat_id, DEFAULT_CATEGORIES_UI))
    rows = []
    for key, label in CATEGORY_LABELS.items():
        mark = "✅ " if key in chosen else "⬜️ "
        rows.append([InlineKeyboardButton(text=f"{mark}{label}", callback_data=f"cat:{key}")])
    rows.append([InlineKeyboardButton(text="Готово", callback_data="cat:done")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


def lead_keyboard(index: int, lead: Lead | None = None) -> InlineKeyboardMarkup:
    rows = []
    tg = lead.tg if lead is not None else None
    if tg is not None and tg.handle:
        # прямая ссылка на канал: одно нажатие вместо копирования @ника
        rows.append([InlineKeyboardButton(text="📣 Открыть Telegram", url=tg.url)])
    elif lead is not None and lead.phone:
        # канала может не быть: тогда единственный путь — звонок.
        # tel: работает на телефоне, на компьютере ссылка просто копируется
        rows.append([InlineKeyboardButton(text="📞 Позвонить", url=f"tel:{lead.phone}")])
    rows.append([
        InlineKeyboardButton(text="✅ Одобрить", callback_data=f"lead:approve:{index}"),
        InlineKeyboardButton(text="❌ Не подходит", callback_data=f"lead:reject:{index}"),
    ])
    rows.append([
        InlineKeyboardButton(text="📨 Отправил вручную", callback_data=f"lead:sent:{index}"),
        InlineKeyboardButton(text="🚫 Не писать", callback_data=f"lead:dnc:{index}"),
    ])
    return InlineKeyboardMarkup(inline_keyboard=rows)


# Кэш текущей выдачи, чтобы кнопки ссылались на конкретные лиды.
current_batch: dict[int, list[Lead]] = {}

# чаты, которые сейчас ждут от нас название города
pending_city: set[int] = set()

# какие категории искали последними: «Следующие лиды» не должны подмешивать
# кафе к баням, если поиск был по баням
active_categories: dict[int, list[str]] = {}

# город последнего поиска: «Следующие лиды» должны добирать тот же город,
# а не подмешивать старые лиды из других городов
active_city: dict[int, str] = {}

# LLM и ассистент (могут быть в режиме без ключа — тогда работает запасная логика)
agent = Agent(storage, build_llm())


def format_lead(lead: Lead, position: str = "") -> str:
    contact_lines = []
    if lead.phone:
        contact_lines.append(f"📞 {lead.phone}")
    tg = lead.tg
    if tg.handle:
        # ссылку даём готовую и кликабельную: владельцу нужно открыть канал,
        # а не копировать юзернейм руками
        badge = {"channel": "канал", "group": "группа", "private": "личный аккаунт",
                 "invite": "закрытая ссылка", "not_found": "не найден",
                 "unknown": "не проверен"}.get(tg.kind, tg.kind)
        contact_lines.append(f"📣 Telegram ({badge}): {tg.at or tg.url}\n{tg.url}")
    if lead.instagram:
        contact_lines.append(f"Instagram: {lead.instagram}")
    if lead.vk:
        contact_lines.append(f"VK: {lead.vk}")
    if lead.email:
        contact_lines.append(f"✉️ {lead.email}")
    if not contact_lines:
        contact_lines.append("⚠️ Контактов нет — только адрес, нужен ручной поиск")

    location = ", ".join(x for x in [lead.city, lead.address] if x) or "адрес не указан"
    category = category_label(lead.category)

    esc = html.escape
    contacts = esc("\n".join(contact_lines))
    body = esc(lead.message)
    return (
        f"{position}<b>{esc(lead.name)}</b>\n"
        f"{esc(category)} · {esc(location)}\n"
        f"Скор: {lead.score} · канал: {lead.best_channel.value}\n"
        f"{contacts}\n\n"
        f"💬 <i>Готовое сообщение:</i>\n{body}"
    )


@dp.message(CommandStart())
async def cmd_start(message: Message) -> None:
    user = message.from_user
    if user:
        log.info("Кто-то запустил бота: id=%s username=@%s name=%s",
                 user.id, user.username, user.full_name)
    if not await guard(message):
        return
    await message.answer(
        "Привет! Я ищу малые бизнесы без сайта и готовлю персональные сообщения.\n\n"
        "Важно: я ничего не рассылаю сам. Вы одобряете лид и отправляете сообщение "
        "вручную — так ваш аккаунт остаётся в безопасности, а люди не жалуются на спам.\n\n"
        "Начните с «🔍 Найти клиентов».",
        reply_markup=main_menu(),
    )


@dp.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(
        "/start — меню\n"
        "/find Москва — найти клиентов в городе\n"
        "/recheck Москва — перепроверить Telegram у уже найденных\n"
        "/ask вопрос — спросить ассистента\n"
        "/reset — очистить историю диалога\n"
        "/stats — статистика базы\n"
        "/csv — выгрузить лиды в файл\n\n"
        "В выдаче бизнесы без сайта, которым можно написать: с публичным "
        "Telegram-каналом или хотя бы с телефоном.\n\n"
        "Telegram и телефон — разные пути. В канал можно написать сразу, "
        "кнопка «Открыть Telegram» ведёт прямо туда. По телефону сначала "
        "согласуйте разговор: кнопка «Позвонить» набирает номер.\n\n"
        "Просто напишите вопрос словами — ассистент ответит.\n"
        "Ответьте (reply) на карточку лида и напишите, что не так — он поправит.\n\n"
        "❗️ Перед отправкой проверьте: у человека должно быть согласие на получение "
        "сообщений, либо вы пишете не рекламу, а личное деловое предложение с возможностью отказа.",
        reply_markup=main_menu(),
    )


@dp.message(F.text == "🗂 Категории")
async def choose_categories(message: Message) -> None:
    if not await guard(message):
        return
    await message.answer("Что искать? Отметьте категории:", reply_markup=categories_menu(message.chat.id))


@dp.callback_query(F.data.startswith("cat:"))
async def toggle_category(call: CallbackQuery) -> None:
    value = call.data.split(":", 1)[1]
    chat_id = call.message.chat.id
    chosen = selected_categories.setdefault(chat_id, list(DEFAULT_CATEGORIES_UI))
    if value == "done":
        labels = [CATEGORY_LABELS[c] for c in chosen]
        await call.message.edit_text("Выбрано: " + ", ".join(labels))
        await call.answer("Сохранено")
        return
    if value in chosen:
        chosen.remove(value)
    else:
        chosen.append(value)
    await call.message.edit_reply_markup(reply_markup=categories_menu(chat_id))
    await call.answer()


async def do_discovery(message: Message, place: str) -> None:
    cats = selected_categories.get(message.chat.id) or DEFAULT_CATEGORIES_UI
    note = await message.answer(f"Ищу в «{place}». Это займёт до минуты…")
    try:
        stats = await asyncio.to_thread(run_discovery, storage, place, cats)
    except Exception as exc:  # noqa: BLE001 — показываем причину пользователю
        log.exception("discovery failed")
        await note.edit_text(f"Не получилось: {exc}\n\nПопробуйте другое название города.")
        return
    # stats.city — каноническое название от геокодера: показываем «Саратов»,
    # а не «Саратове», и по нему же фильтруем выдачу
    active_city[message.chat.id] = stats.city or place
    active_categories[message.chat.id] = category_tag_values(cats)
    # Новый поиск по городу начинаем с лучших лидов, а не с того места,
    # где остановились в прошлый раз.
    storage.reset_shown_for_city(active_city[message.chat.id])
    await note.edit_text(f"Готово по «{stats.city or place}»:\n\n{stats.as_text()}")
    await show_next_leads(message, place=stats.city or place)


@dp.message(F.text == "🔍 Найти клиентов")
async def find_clients(message: Message) -> None:
    if not await guard(message):
        return
    pending_city.add(message.chat.id)
    await message.answer("Напишите город, например: Тюмень или Тюмень, Центральный район")


@dp.message(Command("find"))
async def find_cmd(message: Message) -> None:
    if not await guard(message):
        return
    place = message.text.removeprefix("/find").strip()
    if not place:
        await message.answer("Напишите город: /find Тюмень")
        return
    await do_discovery(message, place)


@dp.message(F.text == "📋 Следующие лиды")
async def next_leads_button(message: Message) -> None:
    city = active_city.get(message.chat.id)
    if not city:
        await message.answer("Сначала поиск: /find Город — потом покажу следующие лиды.")
        return
    await show_next_leads(message, place=city)


async def show_next_leads(message: Message, place: str | None = None, batch_size: int = 5) -> None:
    prepare_messages(storage)
    city = place or active_city.get(message.chat.id)
    if not city:
        await message.answer("Сначала поиск: /find Город — потом покажу лиды.")
        return

    def pick(unseen: bool) -> list[Lead]:
        return storage.find_by_city(
            city, status=LeadStatus.NEW, unseen_only=unseen,
            categories=active_categories.get(message.chat.id) or None,
            contactable_only=True)[:batch_size]

    # Выдаём лиды, которым можно написать: с публичным Telegram-каналом или
    # хотя бы с телефоном. Требовать только канал нельзя — в OSM у российских
    # заведений Telegram почти не указан, и выдача была пустой, хотя в базе
    # лежали сотни лидов с телефонами. Сначала непоказанные; если новые
    # кончились — показываем уже виденные, чтобы не было молчания.
    leads = pick(unseen=True)
    repeat = False
    if not leads:
        leads = pick(unseen=False)
        repeat = True
    if not leads:
        await message.answer(
            f"По городу «{city}» новых лидов с Telegram-каналом нет.\n"
            "Попробуйте другой город или другую категорию — Telegram указан "
            "далеко не у всех бизнесов."
        )
        return

    current_batch[message.chat.id] = leads
    # название берём у самого лида: там каноническая форма («Саратов»), а не падеж
    shown_city = leads[0].city or city
    header = f"Город: {shown_city}. Лиды с Telegram-каналом: {len(leads)} шт. по приоритету."
    if repeat:
        header += "\nЭто уже показанные ранее — новых по городу не осталось."
    await message.answer(
        header + "\nПроверьте сообщение и отправьте вручную. "
        "Если контакт не тот — ответьте (reply) на карточку и напишите, что не так."
    )
    for index, lead in enumerate(leads):
        sent = await message.answer(format_lead(lead, position=f"{index + 1}. "),
                                    parse_mode="HTML", reply_markup=lead_keyboard(index, lead))
        storage.link_message(message.chat.id, sent.message_id, lead.key)
    if not repeat:
        storage.mark_shown([l.key for l in leads])


@dp.callback_query(F.data.startswith("lead:"))
async def handle_lead_action(call: CallbackQuery) -> None:
    _, action, index_raw = call.data.split(":")
    index = int(index_raw)
    batch = current_batch.get(call.message.chat.id, [])
    if index >= len(batch):
        await call.answer("Лид устарел, запросите список заново", show_alert=True)
        return
    lead = batch[index]

    mapping = {
        "approve": (LeadStatus.APPROVED, "Одобрено. Отправьте сообщение вручную."),
        "reject": (LeadStatus.REJECTED, "Отмечено как неподходящее."),
        "sent": (LeadStatus.CONTACTED, "Записал: вы отправили сообщение."),
        "dnc": (LeadStatus.DO_NOT_CONTACT, "Записал: этому больше не писать."),
    }
    status, text = mapping[action]
    storage.set_status(lead.key, status)
    if action == "sent":
        storage.log_send(lead.key, lead.best_channel.value)
    await call.answer(text, show_alert=False)
    await call.message.edit_reply_markup(reply_markup=None)


@dp.message(F.text == "📊 Статистика")
async def stats_button(message: Message) -> None:
    if not await guard(message):
        return
    counts = storage.counts_by_status()
    lines = [f"Всего лидов: {storage.total()}", f"Отправлено сегодня: {storage.sent_today()}"]
    for status in LeadStatus:
        lines.append(f"{status.value}: {counts.get(status.value, 0)}")
    await message.answer("\n".join(lines))


@dp.message(F.text == "📤 Экспорт CSV")
async def export_button(message: Message) -> None:
    if not await guard(message):
        return
    path = settings.db_path.parent / "leads_export.csv"
    export_leads_csv(storage, str(path))
    await message.answer_document(FSInputFile(path), caption=f"Выгрузка лидов, всего в базе: {storage.total()}")


@dp.message(Command("recheck"))
async def recheck_cmd(message: Message) -> None:
    """Перепроверить Telegram у уже сохранённых лидов.

    Нужно после починки проверки каналов: раньше часть контактов помечалась
    «не найден» из-за лимита Telegram, и без перепроверки настоящие каналы
    не появятся в выдаче.
    """
    if not await guard(message):
        return
    city = message.text.removeprefix("/recheck").strip() or None
    limit = 200
    note = await message.answer(
        f"Перепроверяю Telegram{' по ' + city if city else ''} "
        f"(до {limit} контактов). Это займёт пару минут…")
    try:
        checked, channels = await asyncio.to_thread(recheck_telegram, storage, city, limit)
    except Exception as exc:  # noqa: BLE001 — причину показываем владельцу
        log.exception("recheck failed")
        await note.edit_text(f"Не получилось: {exc}")
        return
    if not checked:
        await note.edit_text("Проверять нечего: тип Telegram уже известен у всех лидов.")
        return
    await note.edit_text(
        f"Проверил контактов: {checked}\n"
        f"Из них оказались каналами: {channels}\n\n"
        "Теперь «Следующие лиды» показывают только каналы и дают ссылку.")


@dp.message(Command("ask"))
async def ask_cmd(message: Message) -> None:
    """Явно поговорить с ассистентом."""
    if not await guard(message):
        return
    question = message.text.removeprefix("/ask").strip()
    await talk_to_agent(message, question or "Что сейчас в базе?")


@dp.message(Command("reset"))
async def reset_cmd(message: Message) -> None:
    """Забыть историю диалога."""
    if not await guard(message):
        return
    removed = storage.clear_dialogue(message.chat.id)
    agent.toolbox.last_shown = []
    await message.answer(f"Историю диалога очистил ({removed} сообщ.). Контекст начат заново.")


async def talk_to_agent(message: Message, text: str) -> None:
    """Отправить сообщение ассистенту и показать ответ."""
    context_lead_key = None
    replied = message.reply_to_message
    if replied:
        context_lead_key = storage.lead_key_for_message(message.chat.id, replied.message_id)

    note = await message.answer("Думаю…")
    chat_id = message.chat.id
    task = asyncio.create_task(
        asyncio.to_thread(agent.respond, chat_id, text, context_lead_key)
    )

    # Бесплатная модель иногда зависает. Не заставляем человека ждать вслепую:
    # каждые 15 секунд обновляем статус, а после 120 секунд отвечаем честно.
    waited = 0
    while waited < AGENT_TIMEOUT:
        try:
            answer = await asyncio.wait_for(asyncio.shield(task), timeout=15)
            break
        except asyncio.TimeoutError:
            waited += 15
            await note.edit_text(f"Думаю… уже {waited} секунд. Нейросеть отвечает медленно.")
    else:
        await note.edit_text(
            "Нейросеть не ответила за 2 минуты — видимо, сервис перегружен.\n\n"
            "Пока могу работать по командам: «найди в Саратове», «покажи следующие», "
            "«статистика». Или напишите ещё раз через минуту."
        )
        return

    await note.edit_text(render_answer(answer))


def render_answer(text: str) -> str:
    """Ответ ассистента может содержать что угодно — чистим markdown и режем длину."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # строки-разделители таблиц («|---|---|») и сами таблицы в Telegram
        # выглядят мусором: палки, звёздочки, обратные кавычки
        if re.fullmatch(r"\|?[\s|:\-]+\|?", stripped) and "|" in stripped:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            line = " — ".join(c for c in cells if c)
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
        line = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"\1", line)
        line = line.replace("`", "")
        lines.append(line)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    safe = html.escape(cleaned)
    if len(safe) > 3800:
        safe = safe[:3800] + "…"
    return safe


@dp.errors()
async def on_error(event: ErrorEvent) -> None:
    """Любое исключение в обработчике показываем, а не молчим.

    Без этого падение внутри хендлера aiogram только пишет в лог — а владелец
    на хостинге логов не видит и делает вывод «бот ничего не отвечает».
    """
    log.exception("Ошибка в обработчике", exc_info=event.exception)
    runtime_stats["errors"] += 1
    runtime_stats["last_error"] = (str(event.exception)[:200]
                                   or event.exception.__class__.__name__)
    update = event.update
    message = getattr(update, "message", None)
    if message is None:
        return
    text = str(event.exception)[:300] or event.exception.__class__.__name__
    try:
        await message.answer(
            "Внутренняя ошибка, я её записал в лог:\n"
            f"<code>{html.escape(text)}</code>\n\n"
            "Попробуйте ещё раз или другую команду."
        )
    except Exception:  # noqa: BLE001 — сообщить не вышло, но процесс ронять нельзя
        log.warning("не удалось сообщить об ошибке в чат")


@dp.message(F.text)
async def free_text(message: Message) -> None:
    """Любой текст: либо название города, либо вопрос ассистенту."""
    if not await guard(message):
        return
    text = (message.text or "").strip()
    if not text:
        return

    if message.chat.id in pending_city and not text.endswith("?"):
        pending_city.discard(message.chat.id)
        await do_discovery(message, text)
        return

    pending_city.discard(message.chat.id)
    await talk_to_agent(message, text)


async def main() -> None:
    settings.validate_bot()
    if not settings.allowed_user_ids:
        log.warning("ALLOWED_USER_IDS не задан: бот ответит любому, кто его найдёт.")
    # на хостингах вроде Render нужен открытый порт, иначе сервис считают упавшим
    if settings.port:
        start_health_server(storage, settings.port, stats_payload)
    await _log_startup()
    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)


async def _log_startup() -> None:
    """Печатаем настройки — по логам хостинга сразу видно, что не так.

    Токен и ключи не печатаем, только факт их наличия.
    """
    log.info(
        "Старт: version=%s allowed_user_ids=%s db=%s llm=%s port=%s token=%s",
        (os.getenv("RENDER_GIT_COMMIT") or "dev")[:7],
        settings.allowed_user_ids or "все (список не задан)",
        settings.db_path,
        settings.llm_provider,
        settings.port or "нет",
        "есть" if settings.telegram_bot_token else "НЕТ",
    )


if __name__ == "__main__":
    asyncio.run(main())
