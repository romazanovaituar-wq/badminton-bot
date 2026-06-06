# RallyIQ bot — Dockerfile (замена nixpacks)
# Шаг 1: собираем текущий бот без MediaPipe, проверяем что миграция работает.
# Системные библиотеки (libGL и др.) уже включены — фундамент готов под CV.

FROM python:3.11-slim

# Системные зависимости:
# - ffmpeg: обработка видео
# - libGL/libglib: нужны OpenCV (и MediaPipe потом)
# - gcc/build: на случай сборки пакетов
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала зависимости (кешируется отдельно от кода)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Затем код
COPY . .

# Запуск бота
CMD ["python", "bot.py"]
