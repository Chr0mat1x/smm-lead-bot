# Автодеплой на VPS одной командой.
#
#   bash deploy/install.sh
#
# Скрипт рассчитан на чистый Debian/Ubuntu VPS и делает всё сам:
# ставит python, копирует проект в /opt/smm_bot, создаёт venv,
# ставит зависимости и включает systemd-сервис с автозапуском.
#
# Запускать из корня проекта. Нужен root или sudo.
set -eu

APP_DIR=/opt/smm_bot
SERVICE=smm-bot
SRC_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
    echo "Запустите через sudo: sudo bash deploy/install.sh"
    exit 1
fi

if [ ! -f "$SRC_DIR/.env" ]; then
    echo "Нет .env в $SRC_DIR — сначала заполните токен и настройки."
    exit 1
fi

echo "==> ставлю python и venv"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip ca-certificates

echo "==> копирую проект в $APP_DIR"
mkdir -p "$APP_DIR"
# сохраняем data/ и .env при повторном деплое
cp -r "$SRC_DIR/smm_bot" "$SRC_DIR/scripts" "$SRC_DIR/requirements.txt" "$APP_DIR/"
cp "$SRC_DIR/.env" "$APP_DIR/.env"
mkdir -p "$APP_DIR/data"

echo "==> создаю venv и ставлю зависимости"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "==> включаю systemd-сервис"
cp "$SRC_DIR/deploy/$SERVICE.service" "/etc/systemd/system/$SERVICE.service"
systemctl daemon-reload
systemctl enable "$SERVICE"
systemctl restart "$SERVICE"

sleep 4
echo
echo "==> статус"
systemctl --no-pager -l status "$SERVICE" | head -12
echo
echo "Готово. Логи: journalctl -u $SERVICE -f"
echo "Остановить: systemctl stop $SERVICE"
