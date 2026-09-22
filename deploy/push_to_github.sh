#!/usr/bin/env bash
# Заливка проекта в уже созданный пустой репозиторий GitHub.
#
#   bash deploy/push_to_github.sh smm-lead-bot
#
# Требует, чтобы репозиторий уже существовал (создать: https://github.com/new).
# Секреты не заливаются: .env, data/ и сессии в .gitignore.
set -eu

REPO_NAME="${1:-smm-lead-bot}"
GITHUB_USER="${GITHUB_USER:-Chr0mat1x}"
REMOTE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

cd "$(dirname "$0")/.."

echo "==> проверяю, что секреты не попадут в git"
if ! git check-ignore -q .env; then
    echo "ОШИБКА: .env не игнорируется — остановись, это утечка токена."
    exit 1
fi
if git grep -I -q -E "[0-9]{8,10}:[A-Za-z0-9_-]{35}"; then
    echo "ОШИБКА: в файлах найден похожий на токен текст — не заливаю."
    exit 1
fi
echo "    ок: .env игнорируется, токенов в файлах нет"

if ! git ls-remote "$REMOTE_URL" >/dev/null 2>&1; then
    echo "==> репозиторий недоступен: $REMOTE_URL"
    echo "    создайте пустой репозиторий здесь: https://github.com/new"
    exit 1
fi

echo "==> настраиваю remote origin"
git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"

echo "==> заливаю ветку master"
git push -u origin master

echo
echo "Готово: https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo "Дальше — деплой на Render: deploy/README.md, вариант 1."
