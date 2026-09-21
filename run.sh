#!/usr/bin/env bash
# Запуск бота с перезапуском при падении.
# Остановить: Ctrl+C, либо pkill -f "smm_bot.bot"
set -u
cd "$(dirname "$0")"

if [ ! -f .env ]; then
    echo "Нет файла .env. Скопируйте .env.example в .env и заполните токен."
    exit 1
fi

mkdir -p data
while true; do
    echo "[$(date '+%H:%M:%S')] запускаю бота..."
    python3 -m smm_bot.bot
    code=$?
    echo "[$(date '+%H:%M:%S')] бот завершился с кодом $code, перезапуск через 5 сек"
    sleep 5
done
