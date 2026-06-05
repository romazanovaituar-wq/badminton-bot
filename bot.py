import os
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

BOT_TOKEN       = os.environ.get('BOT_TOKEN', '')
OPENAI_API_KEY  = os.environ.get('OPENAI_API_KEY', '')
PAYMENT_LINK_5  = os.environ.get('PAYMENT_LINK_5', 'https://your-lemonsqueezy.com/5pack')
PAYMENT_LINK_20 = os.environ.get('PAYMENT_LINK_20', 'https://your-lemonsqueezy.com/20pack')

MOTION_THRESHOLD = 12
MAX_FRAMES = 25
LANG, SHIRT_COLOR, OPPONENT_SHIRT, VIDEO = range(4)

user_credits = {}
user_lang = {}
client = OpenAI(api_key=OPENAI_API_KEY)

TEXTS = {
    'ru': {
        'welcome': "Привет, {name}!\n\nBadminton AI Coach - твой AI-тренер.\n\nЗагружаешь видео -> получаешь разбор:\n- Сильные стороны\n- Технические ошибки\n- Тактика\n- Упражнения\n\nБаланс: {credits} анализов\n\n/free - бесплатный анализ\n/analyze - начать\n/buy - купить пакет",
        'help': "Как пользоваться:\n1. /free - 1 бесплатный анализ\n2. /analyze - начать анализ\n3. /buy - купить пакет\n\nФорматы видео:\n- Файл до 50МБ\n- Google Drive ссылка\n- YouTube ссылка",
        'buy_text': "Выбери пакет:\n\n5 анализов - $9.99\n20 анализов - $19.99\n\nПосле оплаты нажми кнопку ниже",
        'buy_btn_5': "5 анализов - $9.99",
        'buy_btn_20': "20 анализов - $19.99",
        'buy_btn_paid': "Я оплатил - добавь анализы",
        'paid_ok': "Добавлено 5 анализов! Баланс: {credits}\n\n/analyze чтобы начать.",
        'free_ok': "Бесплатный анализ добавлен!\n\n/analyze чтобы начать.",
        'free_used': "Бесплатный анализ уже использован.\n\n/buy чтобы купить пакет.",
        'no_credits': "Нет анализов.\n\n/free - первый раз бесплатно\n/buy - купить пакет",
        'ask_shirt': "Начинаем!\n\nОпиши свою одежду в видео:\nНапример: красная футболка черные шорты",
        'ask_opponent': "Понял - {shirt}\n\nОпиши одежду соперника:",
        'ask_video': "Отлично!\n\nОтправь видео:\n1. Файл до 50МБ\n2. Google Drive ссылка\n3. YouTube ссылка\n\n/cancel - отмена",
        'processing': "Получил! Анализирую... 5-10 минут",
        'downloading': "Скачиваю видео...",
        'extracting': "Извлекаю кадры...",
        'analyzing': "Анализирую {n} кадров...",
        'generating': "Генерирую PDF...",
        'done': "Анализ готов, {name}!\n\nОсталось: {credits} анализов\n\n{next_action}",
        'buy_more': "Закончились? /buy",
        'next_analyze': "/analyze - новый анализ",
        'err_download': "Не смог скачать. Попробуй другую ссылку.",
        'err_format': "Не понял формат. Отправь файл или ссылку.",
        'err_frames': "Не удалось извлечь кадры. Попробуй другое видео.",
        'err_general': "Ошибка. /analyze чтобы начать заново.\n{error}",
        'cancelled': "Отменено. /analyze чтобы начать заново.",
        'report_prompt': "Ты опытный тренер по бадминтону.\nПроанализировал {n} кадров. Игрок: {name}, одежда: {shirt}.\n\nКАДРЫ:\n{frames}\n\nНапиши отчёт НА РУССКОМ:\n\n## СИЛЬНЫЕ СТОРОНЫ\n1.\n2.\n3.\n\n## ТЕХНИЧЕСКИЕ ОШИБКИ\n1. [ошибка]: [почему]\n2.\n3.\n\n## ТАКТИКА\n1.\n2.\n\n## УПРАЖНЕНИЯ\n1. [название]: [как делать]\n2.\n3.\n\n## ИТОГ\n[3 предложения]\n\nТолько на основе кадров.",
        'frame_prompt': "Опиши кадр бадминтона. Только факты. Игрок в {shirt}, игнорируй {opp}.\n- Позиция (сетка/середина/задняя)\n- Удар (смэш/лифт/дроп/нет)\n- Ракетка (высоко/низко)\n- Стойка\n- Ошибка если есть\n2 предложения.",
    },
    'kz': {
        'welcome': "Salem, {name}!\n\nBadminton AI Coach - zheke AI zhattyktyrushy.\n\nBeine jukteysin -> taldau alasyn.\n\nBalans: {credits} taldau\n\n/free - tegіn taldau\n/analyze - bastau\n/buy - paket satyp alu",
        'help': "Qalai paydalaný kerek:\n1. /free - 1 tegіn taldau\n2. /analyze - taldaý bastau\n3. /buy - paket satyp alu\n\nBeyne formattary:\n- Fail 50MB deyin\n- Google Drive sіltemesi\n- YouTube sіltemesi",
        'buy_text': "Paket tanda:\n\n5 taldau - $9.99\n20 taldau - $19.99\n\nTolegennен keyin tomengi batyrmanы bas",
        'buy_btn_5': "5 taldau - $9.99",
        'buy_btn_20': "20 taldau - $19.99",
        'buy_btn_paid': "Toledim - taldaulardy qos",
        'paid_ok': "5 taldau qosyldy! Balans: {credits}\n\n/analyze zhaz.",
        'free_ok': "Tegіn taldau qosyldy!\n\n/analyze zhaz.",
        'free_used': "Tegіn taldau buryn paydaanyldý.\n\n/buy zhaz.",
        'no_credits': "Taldaun zhok.\n\n/free - birіnshі ret tegіn\n/buy - paket satyp alu",
        'ask_shirt': "Bastaymyz!\n\nOsы beynedegiі kiіmdі sіpatta:\nMysaly: qyzyл futbolka qara short",
        'ask_opponent': "Tusіndіm - {shirt}\n\nQarsy las kiіmіn sіpatta:",
        'ask_video': "Keremett!\n\nBeynenі zhiber:\n1. Fail 50MB deyin\n2. Google Drive sіltemesi\n3. YouTube sіltemesi\n\n/cancel - boltyrylmaý",
        'processing': "Aldym! Taldaymyn... 5-10 minut",
        'downloading': "Beineni zhukteude...",
        'extracting': "Kadrlardy shygaruda...",
        'analyzing': "{n} kadrdy taldauda...",
        'generating': "PDF zhasalude...",
        'done': "Taldau dayyn, {name}!\n\nQalgan: {credits} taldau\n\n{next_action}",
        'buy_more': "Bitті? /buy",
        'next_analyze': "/analyze - zhana taldau",
        'err_download': "Zhukteу mumkіn bolmady.",
        'err_format': "Format tusinіlmedі.",
        'err_frames': "Kadrlardy shygaru mumkіn bolmady.",
        'err_general': "Qate. /analyze zhazыp qayta bastan.\n{error}",
        'cancelled': "Boltyrylmady. /analyze qayta bastan.",
        'report_prompt': "Sen tazhibelі badminton zhattyktyrushy syn.\nTaldaldy: {n} kadr. Oynaushy: {name}, kiіmі: {shirt}.\n\nKADRLAR:\n{frames}\n\nQAZAQ TІLІNDE esep zhaz:\n\n## KUSHTI ZHAQTARY\n1.\n2.\n3.\n\n## TEHNIKALY QATELER\n1. [qate]: [nege]\n2.\n3.\n\n## TAKTIKA\n1.\n2.\n\n## ZHATTYGULAR\n1. [atauy]: [qalai]\n2.\n3.\n\n## QORYTYNDY\n[3 soilem]\n\nTek kadrlar negіzіnde.",
        'frame_prompt': "Badminton kadryyn sipatta. Tek faktіler. Oynaushy {shirt} kiіngen, {opp} kiіngendі eleme.\n- Pozitsiya\n- Soqqy\n- Rakettka\n- Turys\n- Qate bar bolsa\n2 soilem.",
    },
    'en': {
        'welcome': "Hello, {name}!\n\nBadminton AI Coach - your personal AI trainer.\n\nUpload match video -> get detailed analysis:\n- Strengths\n- Technical mistakes\n- Tactics\n- Drills\n\nBalance: {credits} analyses\n\n/free - free analysis\n/analyze - start\n/buy - buy package",
        'help': "How to use:\n1. /free - 1 free analysis\n2. /analyze - start analysis\n3. /buy - buy package\n\nVideo formats:\n- File up to 50MB\n- Google Drive link\n- YouTube link",
        'buy_text': "Choose package:\n\n5 analyses - $9.99\n20 analyses - $19.99\n\nAfter payment press button below",
        'buy_btn_5': "5 analyses - $9.99",
        'buy_btn_20': "20 analyses - $19.99",
        'buy_btn_paid': "I paid - add analyses",
        'paid_ok': "5 analyses added! Balance: {credits}\n\nType /analyze to start.",
        'free_ok': "Free analysis added!\n\nType /analyze to start.",
        'free_used': "Free analysis already used.\n\nType /buy to purchase.",
        'no_credits': "No analyses left.\n\n/free - first time free\n/buy - buy package",
        'ask_shirt': "Let's start!\n\nDescribe your outfit in this video:\nE.g.: red shirt black shorts",
        'ask_opponent': "Got it - {shirt}\n\nDescribe opponent's outfit:",
        'ask_video': "Great!\n\nSend your video:\n1. File up to 50MB\n2. Google Drive link\n3. YouTube link\n\n/cancel - cancel",
        'processing': "Got it! Analyzing... 5-10 minutes",
        'downloading': "Downloading video...",
        'extracting': "Extracting frames...",
        'analyzing': "Analyzing {n} frames...",
        'generating': "Generating PDF...",
        'done': "Analysis ready, {name}!\n\nLeft: {credits} analyses\n\n{next_action}",
        'buy_more': "Out of analyses? /buy",
        'next_analyze': "/analyze - new analysis",
        'err_download': "Could not download. Try another link.",
        'err_format': "Format not recognized. Send file or link.",
        'err_frames': "Could not extract frames. Try another video.",
        'err_general': "Error. Type /analyze to start over.\n{error}",
        'cancelled': "Cancelled. Type /analyze to start over.",
        'report_prompt': "You are an expert badminton coach.\nAnalyzed {n} frames. Player: {name}, outfit: {shirt}.\n\nFRAMES:\n{frames}\n\nWrite report IN ENGLISH:\n\n## STRENGTHS\n1.\n2.\n3.\n\n## TECHNICAL MISTAKES\n1. [mistake]: [why]\n2.\n3.\n\n## TACTICAL PATTERNS\n1.\n2.\n\n## DRILLS\n1. [name]: [how to do]\n2.\n3.\n\n## SUMMARY\n[3 sentences]\n\nBase on visible frames only.",
        'frame_prompt': "Describe this badminton frame. Facts only. Focus on player in {shirt}, ignore {opp}.\n- Position (net/mid/baseline)\n- Shot type\n- Racket position\n- Stance\n- Any mistake\n2 sentences.",
    }
}

def t(user_id, key, **kwargs):
    lang = user_lang.get(user_id, 'ru')
    text = TEXTS[lang].get(key, key)
    return text.format(**kwargs) if kwargs else text

def download_video(url, dest_path):
    result = subprocess.run([
        'yt-dlp', '-f', 'best[height<=480]',
        '-o', dest_path, '--no-playlist', url
    ], capture_output=True, text=True)
    return os.path.exists(dest_path)

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

def encode_image(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def analyze_video(frames, shirt, opp_shirt, player_name, lang='ru'):
    frame_prompt = TEXTS[lang]['frame_prompt']
    report_prompt = TEXTS[lang]['report_prompt']
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
                {'type': 'text', 'text': frame_prompt.format(shirt=shirt, opp=opp_shirt)}
            ]}]
        )
        desc = resp.choices[0].message.content
        if 'not visible' not in desc.lower():
            descriptions.append(f'Frame {i+1} ({frame["time"]:.0f}s): {desc}')
    if not descriptions:
        return "Could not analyze video - player not visible in frames."
    resp = client.chat.completions.create(
        model='gpt-4o',
        max_tokens=1500,
        messages=[{'role': 'user', 'content': report_prompt.format(
            n=len(descriptions),
            name=player_name,
            shirt=shirt,
            frames='\n'.join(descriptions)
        )}]
    )
    return resp.choices[0].message.content

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

def generate_pdf(report, player_name, frames_count, lang='ru'):
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
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_font('Roboto', '', 10)
    pdf.set_fill_color(245, 248, 255)
    pdf.cell(0, 9, f'  Player: {player_name}   |   Frames: {frames_count}   |   {datetime.now().strftime("%d.%m.%Y")}',
             fill=True, new_x='LMARGIN', new_y='NEXT')
    pdf.ln(5)

    current_section = None
    section_lines = []

    def flush_section():
        if current_section and section_lines:
            pdf.ln(3)
            pdf.set_font('Roboto', 'B', 12)
            pdf.set_fill_color(220, 230, 255)
            pdf.set_text_color(20, 60, 140)
            pdf.cell(0, 9, f'  {current_section}', fill=True, new_x='LMARGIN', new_y='NEXT')
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
            for ln in section_lines:
                pdf.set_font('Roboto', '', 10)
                pdf.multi_cell(185, 6, f'  {ln}' if (ln and ln[0].isdigit()) else ln,
                               new_x='LMARGIN', new_y='NEXT')
                pdf.ln(1)

    for line in report.split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('## '):
            flush_section()
            current_section = line[3:].strip()
            section_lines = []
        else:
            section_lines.append(line)
    flush_section()

    path = f'/tmp/report_{player_name}_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
    pdf.output(path)
    return path

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Русский", callback_data='lang_ru')],
        [InlineKeyboardButton("Kazaksha", callback_data='lang_kz')],
        [InlineKeyboardButton("English",  callback_data='lang_en')],
    ]
    await update.message.reply_text(
        "Выберите язык / Tildi tandanyz / Choose language:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def lang_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    user_lang[user_id] = query.data.replace('lang_', '')
    name = query.from_user.first_name
    credits = user_credits.get(user_id, 0)
    await query.edit_message_text(t(user_id, 'welcome', name=name, credits=credits))

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(t(update.effective_user.id, 'help'))

async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    keyboard = [
        [InlineKeyboardButton(t(uid, 'buy_btn_5'),    url=PAYMENT_LINK_5)],
        [InlineKeyboardButton(t(uid, 'buy_btn_20'),   url=PAYMENT_LINK_20)],
        [InlineKeyboardButton(t(uid, 'buy_btn_paid'), callback_data='paid_5')],
    ]
    await update.message.reply_text(t(uid, 'buy_text'), reply_markup=InlineKeyboardMarkup(keyboard))

async def paid_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    user_credits[uid] = user_credits.get(uid, 0) + 5
    await query.edit_message_text(t(uid, 'paid_ok', credits=user_credits[uid]))

async def free_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if uid not in user_credits:
        user_credits[uid] = 1
        await update.message.reply_text(t(uid, 'free_ok'))
    else:
        await update.message.reply_text(t(uid, 'free_used'))

async def analyze_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if user_credits.get(uid, 0) <= 0:
        await update.message.reply_text(t(uid, 'no_credits'))
        return ConversationHandler.END
    await update.message.reply_text(t(uid, 'ask_shirt'))
    return SHIRT_COLOR

async def get_shirt_color(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    context.user_data['shirt'] = update.message.text
    await update.message.reply_text(t(uid, 'ask_opponent', shirt=update.message.text))
    return OPPONENT_SHIRT

async def get_opponent_shirt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    context.user_data['opponent_shirt'] = update.message.text
    await update.message.reply_text(t(uid, 'ask_video'))
    return VIDEO

async def process_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    shirt = context.user_data.get('shirt', 'unknown')
    opp   = context.user_data.get('opponent_shirt', 'unknown')
    name  = update.effective_user.first_name
    lang  = user_lang.get(uid, 'ru')
    msg   = await update.message.reply_text(t(uid, 'processing'))
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = f'{tmpdir}/video.mp4'
            frames_dir = f'{tmpdir}/frames'
            os.makedirs(frames_dir)
            if update.message.document or update.message.video:
                fo = update.message.document or update.message.video
                tf = await context.bot.get_file(fo.file_id)
                await tf.download_to_drive(video_path)
            elif update.message.text and 'http' in update.message.text:
                await msg.edit_text(t(uid, 'downloading'))
                if not download_video(update.message.text.strip(), video_path):
                    await msg.edit_text(t(uid, 'err_download'))
                    return ConversationHandler.END
            else:
                await msg.edit_text(t(uid, 'err_format'))
                return ConversationHandler.END
            await msg.edit_text(t(uid, 'extracting'))
            frames = extract_frames(video_path, frames_dir, MOTION_THRESHOLD, MAX_FRAMES)
            if len(frames) < 3:
                await msg.edit_text(t(uid, 'err_frames'))
                return ConversationHandler.END
            await msg.edit_text(t(uid, 'analyzing', n=len(frames)))
            report = analyze_video(frames, shirt, opp, name, lang)
            await msg.edit_text(t(uid, 'generating'))
            pdf_path = generate_pdf(report, name, len(frames), lang)
            user_credits[uid] = user_credits.get(uid, 1) - 1
            remaining = user_credits.get(uid, 0)
            next_action = t(uid, 'buy_more') if remaining == 0 else t(uid, 'next_analyze')
            with open(pdf_path, 'rb') as f:
                await update.message.reply_document(
                    document=f,
                    filename=f'BadmintonAI_{name}_{datetime.now().strftime("%d%m%Y")}.pdf',
                    caption=t(uid, 'done', name=name, credits=remaining, next_action=next_action)
                )
            await msg.delete()
    except Exception as e:
        await msg.edit_text(t(uid, 'err_general', error=str(e)))
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(t(update.effective_user.id, 'cancelled'))
    return ConversationHandler.END

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
    app.add_handler(CommandHandler('start',  start))
    app.add_handler(CommandHandler('help',   help_command))
    app.add_handler(CommandHandler('buy',    buy))
    app.add_handler(CommandHandler('free',   free_analysis))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(lang_callback, pattern='^lang_'))
    app.add_handler(CallbackQueryHandler(paid_callback, pattern='^paid_5$'))
    print('Bot started!')
    app.run_polling()

if __name__ == '__main__':
    main()
