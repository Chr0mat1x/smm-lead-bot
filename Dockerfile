FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# зависимости отдельно от кода — слой кешируется при пересборке
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY smm_bot/ ./smm_bot/
COPY scripts/ ./scripts/

# база лидов живёт здесь; на хостинге примонтируйте сюда volume
RUN mkdir -p /app/data
VOLUME ["/app/data"]

# процесс не отдаёт порт, но хостинги требуют живой процесс — health включается через PORT
CMD ["python", "-m", "smm_bot.bot"]
