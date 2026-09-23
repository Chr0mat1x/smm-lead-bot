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
from .tools import ToolBox

log = logging.getLogger("smm_bot.agent")

MAX_STEPS = 5
MAX_HISTORY = 20

SYSTEM_PROMPT = """Ты — ассистент, который помогает находить клиентов на сайты и автоматизацию.
Владельца зовут {your_name}. Он продаёт сайты и автоматизацию малым бизнесам.

Твоя задача:
- находить бизнесы БЕЗ сайта (кафе, бани, салоны, автомойки и подобные);
- показывать карточки лидов и готовые персональные письма;
- когда владелец говорит, что контакт не тот, неподходящий или просит убрать —
  сразу вызывай reject_lead и бери следующий лид;
- когда просит переписать письмо — вызывай set_message с готовым текстом;
- когда просит написать кому-то — не отправляй сам, а покажи текст и напомни,
  что отправляет владелец вручную (авторассылка запрещена и банит аккаунт).

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
    KEYWORDS_SHOW = ("покажи", "следующи", "дальше", "ещё", "еще")
    KEYWORDS_STATS = ("стат", "сколько")
    KEYWORDS_DNC = ("не пиши", "не писать", "черный список", "чёрный список")

    def _fallback(self, chat_id: int, user_text: str, context_lead_key: str | None,
                  reason: str = "") -> str:
        text = user_text.lower()

        if context_lead_key and any(k in text for k in self.KEYWORDS_REJECT):
            self.toolbox.run("reject_lead", {"lead_key": context_lead_key,
                                            "reason": "сказал владелец (без LLM)"})
            follow = self._next_after(context_lead_key)
            return ("Убрал этот контакт — он помечен как неподходящий.\n\n"
                    + (follow or "Следующих подходящих лидов пока нет. Запустите поиск по новому городу."))

        if context_lead_key and any(k in text for k in self.KEYWORDS_DNC):
            self.toolbox.run("do_not_contact", {"lead_key": context_lead_key})
            return "Добавил в чёрный список — этому контакту больше не пишем."

        if any(k in text for k in self.KEYWORDS_FIND):
            city = self._extract_city(user_text)
            if city:
                outcome = self.toolbox.run("search_leads", {"city": city})
                return outcome["result"] + ("\n\n" + (self._next_after(None) or ""))
            return "Напишите город: например «найди клиентов в Екатеринбурге»."

        if any(k in text for k in self.KEYWORDS_STATS):
            return self.toolbox.run("stats", {})["result"]

        if any(k in text for k in self.KEYWORDS_SHOW):
            return self._next_after(context_lead_key) or "Новых подходящих лидов нет."

        if reason:
            log.warning("причина запасного режима: %s", reason)
        return ("Нейросеть сейчас не ответила, поэтому понимаю только простые команды: "
                "«найди в Тюмени», «покажи следующие», «статистика», или ответьте на карточку лида "
                "«не тот контакт».\n\n"
                f"Для разбора у меня сохранена причина: {reason[:300]}\n\n"
                "У бесплатного режима (LLM_PROVIDER=pollinations) ключ не нужен — "
                "проблема на стороне сервиса. Попробуйте написать ещё раз через минуту.")

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
