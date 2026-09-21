"""Живая проверка: агент + бесплатная нейросеть, без ключей.

Запускает реальный сценарий «это не тот контакт» через настоящую модель
Pollinations (GPT-OSS 20B) и проверяет, что лид действительно отклонился.

    python -m scripts.free_llm_demo

Скрипт использует временную базу, вашу рабочую не трогает.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smm_bot.agent import Agent  # noqa: E402
from smm_bot.llm import OpenAICompatibleLLM, POLLINATIONS_ENDPOINT, POLLINATIONS_MODEL  # noqa: E402
from smm_bot.models import Lead, LeadStatus  # noqa: E402
from smm_bot.storage import Storage  # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp()) / "demo.sqlite3"
    storage = Storage(tmp)

    lead = Lead(source="osm", source_id="node/1", name="Кофейня «Пышка»", category="cafe",
                city="Тюмень", telegram="@pyshka", score=80)
    storage.upsert_lead(lead)
    stored = storage.get(lead.key)
    print(f"Лид в базе: {stored.name} (статус {stored.status.value})")

    llm = OpenAICompatibleLLM("", POLLINATIONS_ENDPOINT, POLLINATIONS_MODEL,
                              endpoint=POLLINATIONS_ENDPOINT, timeout=90)
    agent = Agent(storage, llm)

    print("\nОтправляю агенту: «это не тот контакт, это не кофейня» (reply на карточку)")
    answer = agent.respond(chat_id=1, user_text="это не тот контакт, это вообще не кофейня",
                           context_lead_key=stored.key)
    print("\nОтвет ассистента:\n" + answer)

    after = storage.get(stored.key)
    print(f"\nСтатус лида после разговора: {after.status.value}")
    if after.status is LeadStatus.REJECTED:
        print("ИТОГ: лид отклонён автоматически — сценарий работает.")
    else:
        print("ИТОГ: статус не изменился. Модель ответила, но инструмент не вызвала.")
        print("Это нормальный риск бесплатных моделей — см. заметки в README.")
    print(f"\nВременная база: {tmp}")


if __name__ == "__main__":
    main()
