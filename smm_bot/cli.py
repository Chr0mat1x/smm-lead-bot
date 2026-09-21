"""CLI: поиск лидов без запуска бота.

  python -m smm_bot.cli find "Тюмень" --categories cafe,banya,barber
  python -m smm_bot.cli list --limit 10
  python -m smm_bot.cli export --out leads.csv
  python -m smm_bot.cli stats
"""
from __future__ import annotations

import argparse

from .config import settings
from .models import LeadStatus
from .pipeline import export_leads_csv, run_discovery
from .storage import Storage


def main() -> None:
    parser = argparse.ArgumentParser(description="Поиск клиентов без сайта")
    sub = parser.add_subparsers(dest="command", required=True)

    p_find = sub.add_parser("find", help="Найти лиды в городе")
    p_find.add_argument("place", help="Город или район, напр. 'Тюмень'")
    p_find.add_argument("--categories", default="", help="Список через запятую (по умолчанию preset)")

    p_list = sub.add_parser("list", help="Показать лиды из базы")
    p_list.add_argument("--status", default="new")
    p_list.add_argument("--limit", type=int, default=10)

    p_export = sub.add_parser("export", help="Выгрузить лиды в CSV")
    p_export.add_argument("--out", default="data/leads_export.csv")
    p_export.add_argument("--status", default="")
    p_export.add_argument("--limit", type=int, default=500)

    sub.add_parser("stats", help="Статистика базы")

    args = parser.parse_args()
    storage = Storage(settings.db_path)

    if args.command == "find":
        categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None
        stats = run_discovery(storage, args.place, categories)
        print(stats.as_text())

    elif args.command == "list":
        status = LeadStatus(args.status) if args.status else None
        for lead in storage.list_leads(status=status, limit=args.limit):
            contacts = ", ".join(x for x in [lead.phone, lead.telegram, lead.instagram] if x) or "нет контактов"
            print(f"[{lead.score:>4}] {lead.name} ({lead.category}, {lead.city}) — {contacts} — {lead.status.value}")
            if lead.message:
                print("        " + lead.message.split("\n")[0])

    elif args.command == "export":
        status = LeadStatus(args.status) if args.status else None
        path = export_leads_csv(storage, args.out, status=status, limit=args.limit)
        print(f"Сохранено: {path}")

    elif args.command == "stats":
        print(f"Всего: {storage.total()}")
        for status, count in storage.counts_by_status().items():
            print(f"  {status}: {count}")
        print(f"Отправлено сегодня: {storage.sent_today()}")


if __name__ == "__main__":
    main()
