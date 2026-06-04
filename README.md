# Badminton AI Coach Bot

## Файлы
- `bot.py` — основной код бота
- `requirements.txt` — библиотеки
- `Procfile` — команда запуска для Railway

## Как запустить на Railway

### Шаг 1 — Загрузи файлы на GitHub
1. Зайди на github.com
2. Нажми "New repository"
3. Назови: `badminton-bot`
4. Загрузи все 3 файла (bot.py, requirements.txt, Procfile)

### Шаг 2 — Подключи Railway
1. Зайди на railway.app
2. "New Project" → "Deploy from GitHub repo"
3. Выбери репозиторий `badminton-bot`

### Шаг 3 — Добавь переменные окружения
В Railway → Variables добавь:
- `BOT_TOKEN` = твой токен от BotFather
- `OPENAI_API_KEY` = твой ключ OpenAI
- `PAYMENT_LINK_5` = ссылка на 5 анализов из Lemon Squeezy
- `PAYMENT_LINK_20` = ссылка на 20 анализов из Lemon Squeezy

### Шаг 4 — Deploy
Нажми Deploy — бот запустится и будет работать 24/7
