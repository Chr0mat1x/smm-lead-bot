"""Диалоговый цикл ассистента: сообщение -> инструменты -> ответ.

Логика намеренно простая и предсказуемая:

  1. собираем историю диалога + системную инструкцию;
  2. спрашиваем LLM, какой инструмент (если нужно) вызвать;
  3. выполняем инструменты, кладём их результат обратно в диалог;
  4. повторяем, пока модель не ответит текстом (не больше MAX_STEPS раз).

Лимит шагов защищает от зацикливания и от того, что модель бесконечно
«исправляет» одно и то же. При превышении возвращаем то, что успели.

Если ключа LLM нет, работаем в запасном режиме: понимаем простые команды
словами ("найди", "покажи", "отклони", "статистика"), но без свободного диалога.
Это лучше, чем падать посреди переписки.
"""
from __future__ import annotations

import json
import logging
import re

from .config import settings
from .llm import BaseLLM, LLMError, ToolCall
from .models import LeadStatus
from .storage import Storage
from .tools import ToolBox, category_label, contact_hint

log = logging.getLogger("smm_bot.agent")

MAX_STEPS = 5
MAX_HISTORY = 20

SYSTEM_PROMPT = """Ты — ассистент, который помогает находить клиентов на сайты и автоматизацию.
Владельца зовут {your_name}. Он продаёт сайты и автоматизацию малым бизнесам.

Твоя задача:
- находить бизнесы БЕЗ сайта (кафе, бани, салоны, автомойки и подобные);
- показывать только лиды с публичным Telegram-каналом и всегда давать ссылку на него;
- показывать карточки лидов и готовые персональные письма;
- когда владелец говорит, что контакт не тот, неподходящий или просит убрать —
  сразу вызывай reject_lead и бери следующий лид;
- когда просит переписать письмо — вызывай set_message с готовым текстом;
- когда просит написать кому-то — не отправляй сам, а покажи текст и напомни,
  что отправляет владелец вручную (авторассылка запрещена и банит аккаунт).

Каждый лид в выдаче обязан содержать ссылку вида t.me/имя. Если канала нет,
не показывай такой лид и честно скажи, что по этому городу каналов не нашлось.

Правила общения: по-русски, коротко, без канцелярита. Не выдумывай данные —
если чего-то не знаешь, вызови инструмент и посмотри. Если владелец пишет
«второй», «этот», «он» — смотри последнюю выдачу в истории диалога.

Формат ответа: обычный текст, без markdown. Не делай таблиц с «|» — в Telegram
они превращаются в мусор. Лиды перечисляй строками:
  1. Название (категория, адрес) — телефон, скор
Пиши 3-5 лидов за раз, чтобы ответ оставался коротким.

Никогда не обещай массовую рассылку и не предлагай обходить ограничения Telegram."""


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(your_name=settings.your_name)


SUCCESS_WORDS = ("готово", "сделано", "нашёл", "нашел", "найдено", "показываю",
                 "вот список", "выполнено", "обновил")


def _looks_like_success(text: str) -> bool:
    """Похоже ли, что модель отчитывается об успехе, ничего не сделав."""
    low = text.lower()
    return any(word in low for word in SUCCESS_WORDS)


class Agent:
    def __init__(self, storage: Storage, llm: BaseLLM) -> None:
        self.storage = storage
        self.llm = llm
        self.toolbox = ToolBox(storage)

    def respond(self, chat_id: int, user_text: str, context_lead_key: str | None = None) -> str:
        """Главная точка входа: получить ответ ассистента и сохранить историю."""
        hint = ""
        if context_lead_key:
            hint = (f"\n[Контекст: владелец ответил на карточку лида {context_lead_key}. "
                    f"Если он говорит «это не тот», «убери», «не подходит» — "
                    f"относись это именно к лиду {context_lead_key}.]")
        self.storage.add_dialogue(chat_id, "user", user_text + hint, lead_key=context_lead_key or "")

        # Простые команды («найди в Казани», «не тот контакт», «статистика»)
        # выполняем сами: бесплатная модель часто в очереди и отвечает минуту,
        # а эти действия должны работать мгновенно и всегда.
        quick = self.quick_intent(user_text, context_lead_key)
        if quick is not None:
            self.storage.add_dialogue(chat_id, "assistant", quick)
            return quick

        try:
            answer = self._run_llm(chat_id)
        except LLMError as exc:
            log.warning("LLM недоступен, включаю запасной режим: %s", exc)
            answer = self._fallback(chat_id, user_text, context_lead_key, reason=str(exc))
        except Exception:  # noqa: BLE001 — инструмент упал, но диалог ронять нельзя
            log.exception("ошибка при обработке сообщения")
            answer = self._fallback(chat_id, user_text, context_lead_key)

        self.storage.add_dialogue(chat_id, "assistant", answer)
        return answer

    def _run_llm(self, chat_id: int) -> str:
        history = self.storage.get_dialogue(chat_id, limit=MAX_HISTORY)
        messages: list[dict] = [{"role": "system", "content": build_system_prompt()}]
        messages += history
        failed: list[str] = []

        for step in range(MAX_STEPS):
            reply = self.llm.chat(messages, tools=self.toolbox.schemas())

            if not reply.wants_tools:
                text = reply.text.strip()
                # Модель иногда пишет «Готово», хотя инструмент только что упал.
                # Тогда показываем причину сбоя, а не бодрый отчёт.
                if text and failed and _looks_like_success(text):
                    return "Не получилось выполнить поиск. " + failed[-1]
                return text or "Не понял, уточните, что сделать."

            # кладём запрос модели на вызов инструментов в историю
            messages.append({
                "role": "assistant",
                "content": reply.text or None,
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}}
                    for call in reply.tool_calls
                ],
            })

            for call in reply.tool_calls:
                try:
                    outcome = self.toolbox.run(call.name, call.arguments)
                except Exception as exc:  # noqa: BLE001 — поиск может упасть по сети
                    log.exception("инструмент %s упал", call.name)
                    outcome = {"ok": False, "result": f"Инструмент {call.name} не сработал: {exc}"}
                log.info("tool %s(%s) -> ok=%s", call.name, call.arguments, outcome.get("ok"))
                if not outcome.get("ok"):
                    failed.append(str(outcome.get("result", "инструмент не сработал"))[:200])
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.name,
                    "content": outcome.get("result", ""),
                })

        # модель так и не заговорила — просим подвести итог без инструментов
        final = self.llm.chat(messages)
        return final.text.strip() or "Готово. Проверьте базу."

    # ---------- запасной режим без LLM ----------

    KEYWORDS_REJECT = ("не тот", "не подходит", "не клиент", "убери", "удали", "не по адресу",
                       "спам", "ошибка", "накосячил")
    KEYWORDS_FIND = ("найди", "ищи", "поиск", "поищи")
    KEYWORDS_SHOW = ("покажи", "следующ", "дальше", "ещё", "еще")
    KEYWORDS_STATS = ("стат", "сколько")
    KEYWORDS_DNC = ("не пиши", "не писать", "черный список", "чёрный список")

    # Ясные команды распознаём до нейросети. Бесплатная модель то стоит в очереди
    # (429 «Queue full»), то отвечает минуту — и тогда «бот не работает», хотя
    # владельцу нужно всего лишь «найди в Казани». Такие фразы разбираем сами,
    # а нейросеть оставляем для свободного диалога.
    RE_FIND = re.compile(r"^\s*(?:найди|поищи|ищи|поиск)\b", re.I)
    RE_SHOW = re.compile(r"^\s*(?:покажи|следующ|дальше|ещё|еще)\b", re.I)
    # «покажи статистику» начинается как показ, но по смыслу это статистика —
    # поэтому ищем слово в любом месте и проверяем раньше показа
    RE_STATS = re.compile(r"\b(?:стат|сколько)\w*", re.I)

    def quick_intent(self, user_text: str, context_lead_key: str | None = None) -> str | None:
        """Ответ на очевидную команду без обращения к нейросети.

        Возвращает None, если фраза не распознана — тогда вызывающий код идёт
        в обычный диалог с моделью.
        """
        text = (user_text or "").strip()
        low = text.lower()

        # ответ на карточку лида: «это не тот контакт» — самое частое действие
        # владельца, и оно не должно зависеть от доступности нейросети
        if context_lead_key and any(k in low for k in self.KEYWORDS_DNC):
            self.toolbox.run("do_not_contact", {"lead_key": context_lead_key})
            return "Добавил в чёрный список — этому контакту больше не пишем."
        if context_lead_key and any(k in low for k in self.KEYWORDS_REJECT):
            return self._reject_and_next(context_lead_key)

        if self.RE_FIND.match(text):
            city = self._extract_city(text)
            if not city:
                if self.toolbox.active_city:
                    return self._show_current(self.toolbox.active_city)
                return ("Напишите город: например «найди клиентов в Екатеринбурге» "
                        "или «найди кафе с телеграмом в Казани».")
            return self._find_and_show(city, self._extract_categories(text))

        # статистику проверяем раньше показа: «покажи статистику» начинается
        # со слова-показа, но по смыслу это статистика
        if self.RE_STATS.search(text):
            return self.toolbox.run("stats", {})["result"]

        if self.RE_SHOW.match(text):
            if not self.toolbox.active_city:
                return "Сначала поиск: «найди клиентов в Тюмени» — потом покажу лиды."
            return self._show_current(self.toolbox.active_city)

        return None

    def _find_and_show(self, city: str, categories: list[str] | None = None) -> str:
        args: dict = {"city": city}
        if categories:
            args["categories"] = categories
        outcome = self.toolbox.run("search_leads", args)
        return str(outcome.get("result", "Поиск не удался."))

    def _show_current(self, city: str) -> str:
        leads = list(self.storage.find_by_city(
            city, status=LeadStatus.NEW,
            categories=self.toolbox.active_categories or None,
            contactable_only=True))[:3]
        if not leads:
            return (f"По городу «{city}» лидов с контактом нет. "
                    "Попробуйте другой город.")
        lines = [f"{i + 1}. {l.name} ({category_label(l.category)}) — {contact_hint(l)}"
                 for i, l in enumerate(leads)]
        return "Лиды по приоритету:\n" + "\n".join(lines)

    def _reject_and_next(self, lead_key: str) -> str:
        self.toolbox.run("reject_lead", {"lead_key": lead_key,
                                         "reason": "сказал владелец (быстрая команда)"})
        follow = self._next_after(lead_key)
        return ("Убрал этот контакт — помечен как неподходящий.\n\n"
                + (follow or "Следующих каналов в этом городе нет. "
                             "Запустите поиск по новому городу."))

    def _fallback(self, chat_id: int, user_text: str, context_lead_key: str | None,
                  reason: str = "") -> str:
        # Нейросеть недоступна: пробуем понять команду простыми правилами
        quick = self.quick_intent(user_text, context_lead_key)
        if quick is not None:
            return quick
        log.warning("запасной режим, фраза не распознана: %s", reason)
        return ("Нейросеть сейчас не ответила, поэтому понимаю только простые команды: "
                "«найди в Тюмени», «покажи следующие», «статистика», или ответьте на карточку лида "
                "«не тот контакт».\n\n"
                f"Причина: {reason[:300]}\n\n"
                "У бесплатного режима (LLM_PROVIDER=pollinations) ключ не нужен — "
                "проблема на стороне сервиса. Напишите ещё раз через минуту.")


    def _next_after(self, lead_key: str | None) -> str:
        city = self.toolbox.active_city
        if city:
            leads = [l for l in self.storage.find_by_city(city, status=LeadStatus.NEW,
                                                          unseen_only=True) if l.reachable][:3]
            if not leads:
                return ""
        else:
            leads = [l for l in self.storage.list_leads(status=LeadStatus.NEW, limit=3)
                     if l.reachable]
        if not leads:
            return ""
        lead = leads[0]
        return f"Следующий лид: {lead.name} ({lead.category}, {lead.city}), ключ {lead.key}."

    @staticmethod
    def _extract_city(text: str) -> str | None:
        match = re.search(r"(?:в|по)\s+([А-ЯЁ][\w\-]+(?:\s+[А-ЯЁ][\w\-]+)?)", text)
        if match:
            return match.group(1)
        words = [w for w in re.findall(r"[А-ЯЁ][а-яё\-]{2,}", text)]
        return words[0] if words else None

    @staticmethod
    def _extract_categories(text: str) -> list[str] | None:
        """Понять из фразы, что именно искать: «кафе», «бани», «барбершоп».

        Без этого «найди кафе в Казани» возвращало салоны красоты: слово
        «кафе» игнорировалось, и срабатывал список категорий по умолчанию.
        """
        from .osm import CATEGORY_ALIASES, CATEGORY_PRESETS

        words = re.findall(r"[а-яёa-z_]+", text.lower())
        found: list[str] = []
        for word in words:
            preset = CATEGORY_ALIASES.get(word) or (word if word in CATEGORY_PRESETS else "")
            if preset and preset not in found:
                found.append(preset)
        return found or None
