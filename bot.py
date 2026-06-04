import os
import asyncio
import tempfile
import subprocess
import base64
import urllib.request
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (Application, CommandHandler, MessageHandler,
                          CallbackQueryHandler, ContextTypes,
                          filters, ConversationHandler)
from openai import OpenAI
from fpdf import FPDF
import cv2

# ==================================================
# НАСТРОЙКИ — берутся из переменных окружения Railway
# ==================================================
BOT_TOKEN      = os.environ.get('BOT_TOKEN', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
PAYMENT_LINK_5  = os.environ.get('PAYMENT_LINK_5', 'https://your-lemonsqueezy.com/5pack')
PAYMENT_LINK_20 = os.environ.get('PAYMENT_LINK_20', 'https://your-lemonsqueezy.com/20pack')

MOTION_THRESHOLD = 12
MAX_FRAMES       = 25

# Состояния диалога
SHIRT_COLOR, OPPONENT_SHIRT, VIDEO = range(3)

# База пользователей (в памяти)
user_credits = {}

client = OpenAI(api_key=OPENAI_API_KEY)

# ==================================================
# СКАЧИВАНИЕ ВИДЕО
# ==================================================
def download_video(url, dest_path):
    result = subprocess.run([
        'yt-dlp', '-f', 'best[height<=480]',
        '-o', dest_path, '--no-playlist', url
    ], capture_output=True, text=True)
    return os.path.exists(dest_path)

# ==================================================
# НАРЕЗКА КАДРОВ
# ==================================================
def extract_frames(video_path, output_dir, threshold=12, max_frames=25):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    prev_gray = None
    saved = []
    idx = 0
    last_saved = -30

    while len(saved) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % 5 != 0:
            idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray).mean()
            if diff > threshold and (idx - last_saved) > fps:
                path = f'{output_dir}/frame_{len(saved):03d}.jpg'
                h, w = frame.shape[:2]
                if w > 800:
                    frame = cv2.resize(frame, (800, int(h * 800/w)))
                cv2.imwrite(path, frame)
                saved.append({'path': path, 'time': idx / fps})
                last_saved = idx

        prev_gray = gray
        idx += 1

    cap.release()
    return saved

# ==================================================
# AI АНАЛИЗ
# ==================================================
def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def analyze_video(frames, player_shirt, opponent_shirt, player_name='Player'):
    descriptions = []
    for i, frame in enumerate(frames):
        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            max_tokens=150,
            messages=[{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {
                    'url': f'data:image/jpeg;base64,{encode_image(frame["path"])}',
                    'detail': 'low'
                }},
                {'type': 'text', 'text': f'''Analyze this badminton frame.
Focus ONLY on player wearing {player_shirt}.
Ignore player wearing {opponent_shirt}.
Player may be on any side of court (they switch sides between sets).
Describe:
- Position (net/mid/baseline)
- Shot type (smash/clear/drop/lift/serve/moving/none)
- Racket position (high/low/mid-swing)
- Stance (bent knees/upright/jumping/lunging)
- Any visible technical mistake
If target player not clearly visible say "not visible". Max 2 sentences.'''}
            ]}]
        )
        desc = resp.choices[0].message.content
        if 'not visible' not in desc.lower():
            descriptions.append(f'Frame {i+1} ({frame["time"]:.0f}s): {desc}')

    if not descriptions:
        return "Не удалось проанализировать видео — игрок не был виден в кадрах."

    frames_text = '\n'.join(descriptions)
    resp = client.chat.completions.create(
        model='gpt-4o',
        max_tokens=1500,
        messages=[{'role': 'user', 'content': f'''You are an expert badminton coach with 10 years experience.
You analyzed {len(descriptions)} frames. Player: {player_name}, outfit: {player_shirt}.

FRAME DESCRIPTIONS:
{frames_text}

Write a detailed coaching report IN RUSSIAN using this structure:

## СИЛЬНЫЕ СТОРОНЫ
1. [конкретное наблюдение из кадров]
2. [конкретное наблюдение]
3. [конкретное наблюдение]

## ТЕХНИЧЕСКИЕ ОШИБКИ
1. [ошибка]: [почему это проблема]
2. [ошибка]: [объяснение]
3. [ошибка]: [объяснение]

## ТАКТИКА
1. [паттерн]: [как влияет на игру]
2. [паттерн]: [описание]

## УПРАЖНЕНИЯ
1. [название]: [как делать — 2 предложения]
2. [название]: [описание]
3. [название]: [описание]

## ИТОГ
[3 предложения: общий уровень, главное что исправить, мотивация]

Base every point ONLY on visible frames. If unsure write "недостаточно данных".'''}]
    )
    return resp.choices[0].message.content

# ==================================================
# ГЕНЕРАЦИЯ PDF
# ==================================================
def download_fonts():
    if not os.path.exists('/tmp/Roboto-Regular.ttf'):
        urllib.request.urlretrieve(
            'https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf',
            '/tmp/Roboto-Regular.ttf'
        )
    if not os.path.exists('/tmp/Roboto-Bold.ttf'):
        urllib.request.urlretrieve(
            'https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Bold.ttf',
            '/tmp/Roboto-Bold.ttf'
        )

def generate_pdf(report, player_name, frames_count):
    download_fonts()

    class Report(FPDF):
        def header(self):
            self.set_font('Roboto', 'B', 15)
            self.set_fill_color(20, 60, 140)
            self.set_text_color(255, 255, 255)
            self.cell(0, 14, '  BADMINTON AI COACH', fill=True, new_x='LMARGIN', new_y='NEXT')
            self.set_text_color(0, 0, 0)
            self.ln(3)

        def footer(self):
            self.set_y(-15)
            self.set_font('Roboto', '', 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f'AI Report | {datetime.now().strftime("%d.%m.%Y")} | Page {self.page_no()}', align='C')

    pdf = Report()
    pdf.add_font('Roboto', '', '/tmp/Roboto-Regular.ttf')
    pdf.add_font('Roboto', 'B', '/tmp/Roboto-Bold.ttf')
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    pdf.set_font('Roboto', '', 10)
    pdf.set_fill_color(245, 248, 255)
    pdf.cell(0, 9, f'  Игрок: {player_name}   |   Кадров: {frames_count}   |   {datetime.now().strftime("%d.%m.%Y")}',
             fill=True, new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)

    for line in report.split('\n'):
        line = line.strip()
        if not line:
            pdf.ln(2)
        elif line.startswith('## '):
            pdf.ln(3)
            pdf.set_font('Roboto', 'B', 12)
            pdf.set_fill_color(220, 230, 255)
            pdf.set_text_color(20, 60, 140)
            pdf.cell(0, 9, f'  {line[3:]}', fill=True, new_x='LMARGIN', new_y='NEXT')
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
        elif line and line[0].isdigit():
            pdf.set_font('Roboto', '', 10)
            pdf.multi_cell(190, 6, f'  {line}')
            pdf.ln(1)
        else:
            pdf.set_font('Roboto', '', 10)
            pdf.multi_cell(190, 6, line)

    path = f'/tmp/report_{player_name}_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
    pdf.output(path)
    return path

# ==================================================
# HANDLERS
# ==================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    credits = user_credits.get(user_id, 0)
    name = update.effective_user.first_name
    await update.message.reply_text(
        f'👋 Привет, {name}!\n\n'
        f'🏸 *Badminton AI Coach* — твой персональный AI-тренер.\n\n'
        f'Загружаешь видео матча → получаешь детальный разбор:\n'
        f'• Сильные стороны\n'
        f'• Технические ошибки\n'
        f'• Тактические паттерны\n'
        f'• Конкретные упражнения\n\n'
        f'💳 Твой баланс: *{credits} анализов*\n\n'
        f'/free — получи 1 бесплатный анализ\n'
        f'/analyze — начать анализ\n'
        f'/buy — купить пакет\n'
        f'/help — помощь',
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        '🏸 *Как пользоваться:*\n\n'
        '1. /free — 1 бесплатный анализ\n'
        '2. /analyze — начать анализ видео\n'
        '3. /buy — купить пакет анализов\n\n'
        '*Форматы видео:*\n'
        '• Файл до 50МБ прямо в чат\n'
        '• Google Drive ссылка (любой размер)\n'
        '• YouTube ссылка\n\n'
        '*Совет:* для лучшего анализа опиши одежду подробно\n'
        'Например: "белая футболка чёрные шорты"',
        parse_mode='Markdown'
    )

async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton('🎯 5 анализов — $9.99', url=PAYMENT_LINK_5)],
        [InlineKeyboardButton('🚀 20 анализов — $19.99', url=PAYMENT_LINK_20)],
        [InlineKeyboardButton('✅ Я оплатил — добавь анализы', callback_data='paid_5')]
    ]
    await update.message.reply_text(
        '💳 *Выбери пакет:*\n\n'
        '🎯 5 анализов — $9.99\n'
        '🚀 20 анализов — $19.99\n\n'
        'После оплаты нажми кнопку ниже 👇',
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode='Markdown'
    )

async def paid_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    user_credits[user_id] = user_credits.get(user_id, 0) + 5
    await query.edit_message_text(
        f'✅ Спасибо! Добавлено 5 анализов.\n'
        f'Баланс: {user_credits[user_id]}\n\n'
        f'Напиши /analyze чтобы начать.'
    )

async def free_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in user_credits:
        user_credits[user_id] = 1
        await update.message.reply_text(
            '🎁 Бесплатный анализ добавлен!\n\nНапиши /analyze чтобы начать.'
        )
    else:
        await update.message.reply_text(
            '❌ Бесплатный анализ уже был использован.\n\nНапиши /buy чтобы купить пакет.'
        )

async def analyze_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_credits.get(user_id, 0) <= 0:
        await update.message.reply_text(
            '❌ У тебя нет анализов.\n\n'
            '🎁 Первый раз? Напиши /free\n'
            '💳 Купить пакет: /buy'
        )
        return ConversationHandler.END

    await update.message.reply_text(
        '🏸 Начинаем!\n\n'
        '*Опиши свою одежду в этом видео:*\n\n'
        'Например:\n'
        '• красная футболка чёрные шорты\n'
        '• белая майка синие шорты\n'
        '• тёмно-синяя форма',
        parse_mode='Markdown'
    )
    return SHIRT_COLOR

async def get_shirt_color(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['shirt'] = update.message.text
    await update.message.reply_text(
        f'✅ Понял — *{update.message.text}*\n\n'
        f'Теперь опиши одежду *соперника:*',
        parse_mode='Markdown'
    )
    return OPPONENT_SHIRT

async def get_opponent_shirt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['opponent_shirt'] = update.message.text
    await update.message.reply_text(
        '✅ Отлично!\n\n'
        '📹 *Отправь видео одним из способов:*\n\n'
        '1️⃣ Файл прямо сюда (до 50МБ)\n\n'
        '2️⃣ Google Drive ссылка:\n'
        '   • Загрузи на drive.google.com\n'
        '   • Поделиться → Все у кого есть ссылка\n'
        '   • Скопируй и отправь\n\n'
        '3️⃣ YouTube ссылка\n\n'
        '⚠️ Для отмены напиши /cancel',
        parse_mode='Markdown'
    )
    return VIDEO

async def process_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    shirt = context.user_data.get('shirt', 'unknown color')
    opponent_shirt = context.user_data.get('opponent_shirt', 'unknown color')
    name = update.effective_user.first_name

    msg = await update.message.reply_text('⏳ Получил! Анализирую видео... это займёт 5-10 минут 🎯')

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = f'{tmpdir}/video.mp4'
            frames_dir = f'{tmpdir}/frames'
            os.makedirs(frames_dir)

            # Скачиваем видео
            if update.message.document or update.message.video:
                file_obj = update.message.document or update.message.video
                tg_file = await context.bot.get_file(file_obj.file_id)
                await tg_file.download_to_drive(video_path)
            elif update.message.text and 'http' in update.message.text:
                await msg.edit_text('⏳ Скачиваю видео по ссылке...')
                success = download_video(update.message.text.strip(), video_path)
                if not success:
                    await msg.edit_text('❌ Не смог скачать видео. Попробуй другую ссылку или отправь файл напрямую.')
                    return ConversationHandler.END
            else:
                await msg.edit_text('❌ Не понял формат. Отправь файл или ссылку.')
                return ConversationHandler.END

            await msg.edit_text('⏳ Извлекаю ключевые кадры...')
            frames = extract_frames(video_path, frames_dir, MOTION_THRESHOLD, MAX_FRAMES)

            if len(frames) < 3:
                await msg.edit_text('❌ Не удалось извлечь кадры. Попробуй другое видео.')
                return ConversationHandler.END

            await msg.edit_text(f'⏳ Анализирую {len(frames)} кадров с помощью AI...')
            report = analyze_video(frames, shirt, opponent_shirt, name)

            await msg.edit_text('⏳ Генерирую PDF отчёт...')
            pdf_path = generate_pdf(report, name, len(frames))

            # Списываем кредит
            user_credits[user_id] = user_credits.get(user_id, 1) - 1
            remaining = user_credits.get(user_id, 0)

            with open(pdf_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=f'BadmintonAI_{name}_{datetime.now().strftime("%d%m%Y")}.pdf',
                    caption=f'✅ Анализ готов, {name}!\n\n'
                            f'💳 Осталось анализов: {remaining}\n\n'
                            f'{"💡 Закончились анализы? /buy" if remaining == 0 else "/analyze — новый анализ"}'
                )
            await msg.delete()

    except Exception as e:
        await msg.edit_text(
            f'❌ Произошла ошибка. Попробуй ещё раз.\n\n'
            f'Напиши /analyze чтобы начать заново.\n\n'
            f'Ошибка: {str(e)}'
        )

    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('❌ Отменено. Напиши /analyze чтобы начать заново.')
    return ConversationHandler.END

# ==================================================
# ЗАПУСК
# ==================================================
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler('analyze', analyze_start)],
        states={
            SHIRT_COLOR:    [MessageHandler(filters.TEXT & ~filters.COMMAND, get_shirt_color)],
            OPPONENT_SHIRT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_opponent_shirt)],
            VIDEO: [
                MessageHandler(filters.Document.ALL | filters.VIDEO, process_video),
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_video)
            ],
        },
        fallbacks=[CommandHandler('cancel', cancel)]
    )

    app.add_handler(CommandHandler('start', start))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('buy', buy))
    app.add_handler(CommandHandler('free', free_analysis))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(paid_callback, pattern='paid_5'))

    print('✅ Бот запущен!')
    app.run_polling()

if __name__ == '__main__':
    main()
