"""
RallyIQ — AI-тренер по бадминтону в Telegram.
Production-ready версия с PostgreSQL, защитой от блокировок и многоязычностью.
"""
import os
import asyncio
import logging
import tempfile
import base64
import shutil
from datetime import datetime

from telegram import (Update, InlineKeyboardButton, InlineKeyboardMarkup,
                      ReplyKeyboardRemove)
from telegram.ext import (Application, CommandHandler, MessageHandler,
                          CallbackQueryHandler, ContextTypes,
                          filters, ConversationHandler)
from openai import OpenAI
from fpdf import FPDF
import cv2

import db
from texts import t, TEXTS

# ==================================================
# КОНФИГУРАЦИЯ
# ==================================================
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("rallyiq")

BOT_TOKEN       = os.environ.get("BOT_TOKEN", "")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
PAYMENT_LINK_5  = os.environ.get("PAYMENT_LINK_5", "https://example.com/5")
PAYMENT_LINK_20 = os.environ.get("PAYMENT_LINK_20", "https://example.com/20")

MOTION_THRESHOLD = 12
MAX_FRAMES       = 25
ANALYSIS_TIMEOUT = 600  # 10 минут максимум на анализ
MAX_VIDEO_MB     = 200
ADMIN_ID         = 942577691  # Telegram ID администратора


def _is_admin(user_id: int) -> bool:
    return user_id == ADMIN_ID

IDENTIFY, SHIRT_COLOR, POSITION, VIDEO = range(4)

client = OpenAI(api_key=OPENAI_API_KEY)

# Защита от двойного запуска: кто сейчас обрабатывает видео
processing_users: set[int] = set()


# ==================================================
# КЛАВИАТУРЫ
# ==================================================

def _safe_url(url: str) -> str | None:
    """Возвращает URL только если он валидный http(s), иначе None."""
    url = (url or "").strip()
    if url.startswith("http://") or url.startswith("https://"):
        # Отсекаем мусор вроде пробелов и скобок
        if " " not in url and "(" not in url:
            return url
    return None

def main_keyboard(lang: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "btn_analyze"), callback_data="go_analyze")],
        [InlineKeyboardButton(t(lang, "btn_buy"), callback_data="go_buy")],
    ])


def buy_keyboard(lang: str) -> InlineKeyboardMarkup:
    rows = []
    url5 = _safe_url(PAYMENT_LINK_5)
    url20 = _safe_url(PAYMENT_LINK_20)
    if url5:
        rows.append([InlineKeyboardButton(t(lang, "btn_5"), url=url5)])
    if url20:
        rows.append([InlineKeyboardButton(t(lang, "btn_20"), url=url20)])
    if not rows:
        # Платёжные ссылки ещё не настроены — показываем заглушку
        rows.append([InlineKeyboardButton("⏳ Payment coming soon", callback_data="noop")])
    return InlineKeyboardMarkup(rows)


# ==================================================
# ВИДЕО: скачивание и нарезка (выполняются в executor)
# ==================================================
def _download_video_sync(url: str, dest_path: str) -> bool:
    """
    Скачивает видео через Python-модуль yt-dlp (надёжнее чем subprocess,
    т.к. не зависит от наличия бинарника в PATH).
    Поддерживает YouTube, Google Drive и прямые ссылки.
    """
    try:
        import yt_dlp
    except ImportError:
        logger.error("yt-dlp не установлен")
        return False

    # Google Drive ссылки обрабатываем отдельно через gdown-подобную логику
    if "drive.google.com" in url:
        return _download_gdrive(url, dest_path)

    ydl_opts = {
        "format": "best[height<=480]/best",
        "outtmpl": dest_path,
        "noplaylist": True,
        "max_filesize": MAX_VIDEO_MB * 1024 * 1024,
        "quiet": True,
        "no_warnings": True,
        # Обход части блокировок YouTube для серверов
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error("yt-dlp ошибка: %s", e)
        return False
    return os.path.exists(dest_path)


def _download_gdrive(url: str, dest_path: str) -> bool:
    """Скачивает файл с Google Drive по публичной ссылке."""
    import re
    import urllib.request
    # Извлекаем file id из разных форматов ссылок
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if not m:
        return False
    file_id = m.group(1)
    direct = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        req = urllib.request.Request(direct, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp, open(dest_path, "wb") as out:
            out.write(resp.read())
    except Exception as e:
        logger.error("Google Drive ошибка: %s", e)
        return False
    return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0


def _save_frame(frame, output_dir: str, index: int, time_s: float) -> dict:
    """Сохраняет кадр с ресайзом и возвращает метаданные."""
    h, w = frame.shape[:2]
    if w > 800:
        frame = cv2.resize(frame, (800, int(h * 800 / w)))
    path = f"{output_dir}/frame_{index:03d}.jpg"
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"path": path, "time": time_s}


def _extract_frames_sync(video_path: str, output_dir: str) -> list[dict]:
    """
    Надёжное извлечение кадров с трёхуровневым фоллбэком:
    1) кадры с движением (умный выбор)
    2) равномерно по времени (если движения мало)
    3) любые доступные кадры (если видео короткое/проблемное)
    Гарантированно не возвращает пустоту если в видео есть кадры.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # --- Уровень 1: motion detection ---
    motion: list[dict] = []
    prev_gray = None
    idx = 0
    last_saved = -fps
    while len(motion) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray).mean()
                if diff > MOTION_THRESHOLD and (idx - last_saved) > fps:
                    motion.append(_save_frame(frame, output_dir, len(motion), idx / fps))
                    last_saved = idx
            prev_gray = gray
        idx += 1

    if len(motion) >= 8:
        cap.release()
        logger.info("Извлечено %d кадров (motion)", len(motion))
        return motion

    # --- Уровень 2: равномерно по видео ---
    # Чистим то что насобирал уровень 1
    for fobj in motion:
        try:
            os.remove(fobj["path"])
        except OSError:
            pass

    uniform: list[dict] = []
    if total > 0:
        step = max(1, total // MAX_FRAMES)
        for i in range(0, total, step):
            if len(uniform) >= MAX_FRAMES:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                uniform.append(_save_frame(frame, output_dir, len(uniform), i / fps))
    if len(uniform) >= 3:
        cap.release()
        logger.info("Извлечено %d кадров (uniform)", len(uniform))
        return uniform

    # --- Уровень 3: берём вообще всё что есть ---
    for fobj in uniform:
        try:
            os.remove(fobj["path"])
        except OSError:
            pass

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    any_frames: list[dict] = []
    idx = 0
    while len(any_frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % 10 == 0:
            any_frames.append(_save_frame(frame, output_dir, len(any_frames), idx / fps))
        idx += 1
    cap.release()
    logger.info("Извлечено %d кадров (fallback)", len(any_frames))
    return any_frames


# ==================================================
# AI АНАЛИЗ (выполняется в executor)
# ==================================================
def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _analyze_sync(frames: list[dict], target: str,
                  name: str, lang: str) -> str:
    frame_prompt = t(lang, "frame_prompt", target=target)
    descriptions: list[str] = []

    for i, frame in enumerate(frames):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=150,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{_encode_image(frame['path'])}",
                    "detail": "low",
                }},
                {"type": "text", "text": frame_prompt},
            ]}],
        )
        desc = resp.choices[0].message.content
        if "not visible" not in desc.lower():
            descriptions.append(f"Frame {i+1} ({frame['time']:.0f}s): {desc}")

    if not descriptions:
        return "Could not analyze — player not visible in frames."

    report_prompt = t(lang, "report_prompt",
                      n=len(descriptions), name=name, target=target,
                      frames="\n".join(descriptions))
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=[{"role": "user", "content": report_prompt}],
    )
    return resp.choices[0].message.content


# ==================================================
# PDF
# ==================================================
FONT_REG = "/tmp/Roboto-Regular.ttf"
FONT_BLD = "/tmp/Roboto-Bold.ttf"


def _ensure_fonts() -> None:
    import urllib.request
    if not os.path.exists(FONT_REG):
        urllib.request.urlretrieve(
            "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Regular.ttf",
            FONT_REG)
    if not os.path.exists(FONT_BLD):
        urllib.request.urlretrieve(
            "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Bold.ttf",
            FONT_BLD)


def _generate_pdf_sync(report: str, name: str, frames_count: int,
                       lang: str, out_path: str) -> str:
    _ensure_fonts()

    class Report(FPDF):
        def header(self):
            self.set_font("Roboto", "B", 15)
            self.set_fill_color(20, 60, 140)
            self.set_text_color(255, 255, 255)
            self.cell(0, 14, "  RALLYIQ — AI BADMINTON COACH",
                      fill=True, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.ln(3)

        def footer(self):
            self.set_y(-15)
            self.set_font("Roboto", "", 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10,
                      f"RallyIQ | {datetime.now().strftime('%d.%m.%Y')} | Page {self.page_no()}",
                      align="C")

    pdf = Report()
    pdf.add_font("Roboto", "", FONT_REG)
    pdf.add_font("Roboto", "B", FONT_BLD)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    pdf.set_font("Roboto", "", 10)
    pdf.set_fill_color(245, 248, 255)
    pdf.cell(0, 9,
             f"  Player: {name}   |   Frames: {frames_count}   |   {datetime.now().strftime('%d.%m.%Y')}",
             fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    current = None
    buffer: list[str] = []

    def flush():
        if current and buffer:
            pdf.ln(3)
            pdf.set_font("Roboto", "B", 12)
            pdf.set_fill_color(220, 230, 255)
            pdf.set_text_color(20, 60, 140)
            pdf.cell(0, 9, f"  {current}", fill=True,
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)
            for ln in buffer:
                pdf.set_font("Roboto", "", 10)
                txt = f"  {ln}" if (ln and ln[0].isdigit()) else ln
                pdf.multi_cell(185, 6, txt, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)

    for line in report.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            flush()
            current = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    flush()

    # Дисклеймер
    pdf.ln(6)
    pdf.set_font("Roboto", "", 8)
    pdf.set_text_color(150, 150, 150)
    disclaimer = {
        "ru": "Отчёт сгенерирован AI на основе кадров видео и может содержать неточности. Не заменяет очного тренера.",
        "kz": "Есеп бейне кадрлары негізінде AI арқылы жасалған, дәл болмауы мүмкін. Жаттықтырушыны алмастырмайды.",
        "en": "This report is AI-generated from video frames and may contain inaccuracies. Not a substitute for a real coach.",
    }.get(lang, "")
    pdf.multi_cell(185, 4, disclaimer, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.output(out_path)
    return out_path


# ==================================================
# HANDLERS — команды
# ==================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    u = update.effective_user
    db.ensure_user(u.id, u.username, u.first_name)
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru")],
        [InlineKeyboardButton("🇰🇿 Қазақша", callback_data="lang_kz")],
        [InlineKeyboardButton("🇬🇧 English", callback_data="lang_en")],
    ])
    # Убираем застрявшую старую клавиатуру
    await update.message.reply_text("🏸 RallyIQ", reply_markup=ReplyKeyboardRemove())
    await update.message.reply_text(
        TEXTS["ru"]["choose_lang"], reply_markup=keyboard)


async def lang_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    lang = q.data.replace("lang_", "")
    db.ensure_user(q.from_user.id, q.from_user.username, q.from_user.first_name)
    db.set_lang(q.from_user.id, lang)
    credits = db.get_credits(q.from_user.id)
    await q.edit_message_text(
        t(lang, "welcome", name=q.from_user.first_name, credits=credits))
    await q.message.reply_text("👇", reply_markup=main_keyboard(lang))


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = db.get_lang(update.effective_user.id)
    await update.message.reply_text(t(lang, "help"))


async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = db.get_lang(uid)
    await update.message.reply_text(
        t(lang, "balance", credits=db.get_credits(uid)),
        reply_markup=main_keyboard(lang))


async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = db.get_lang(update.effective_user.id)
    await update.message.reply_text(
        t(lang, "buy_text"), reply_markup=buy_keyboard(lang))


async def buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    await q.message.reply_text(t(lang, "buy_text"), reply_markup=buy_keyboard(lang))


async def paid_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    new_balance = db.add_credits(q.from_user.id, 5)
    await q.edit_message_text(t(lang, "paid_ok", credits=new_balance))


async def free_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = db.get_lang(uid)
    db.ensure_user(uid, update.effective_user.username, update.effective_user.first_name)
    if db.grant_free(uid):
        await update.message.reply_text(
            t(lang, "free_ok"), reply_markup=main_keyboard(lang))
    else:
        await update.message.reply_text(
            t(lang, "free_used"), reply_markup=buy_keyboard(lang))


# ==================================================
# HANDLERS — диалог анализа
# ==================================================
async def analyze_entry_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    return await _start_analysis_flow(update.effective_user, update.message, context)


async def analyze_entry_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    return await _start_analysis_flow(q.from_user, q.message, context)


async def _start_analysis_flow(user, message, context):
    uid = user.id
    lang = db.get_lang(uid)
    db.ensure_user(uid, user.username, user.first_name)

    if uid in processing_users:
        await message.reply_text(t(lang, "busy"))
        return ConversationHandler.END

    # Админу анализы всегда бесплатны
    if not _is_admin(uid):
        # Бесплатный анализ для новичка, либо проверка баланса
        if db.get_credits(uid) <= 0:
            if db.grant_free(uid):
                pass  # выдали бесплатный
            else:
                await message.reply_text(
                    t(lang, "no_credits"), reply_markup=buy_keyboard(lang))
                return ConversationHandler.END

    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "btn_by_color"), callback_data="id_color")],
        [InlineKeyboardButton(t(lang, "btn_by_position"), callback_data="id_position")],
        [InlineKeyboardButton(t(lang, "btn_dont_know"), callback_data="id_both")],
    ])
    await message.reply_text(t(lang, "ask_identify"), reply_markup=kb)
    return IDENTIFY


async def identify_color(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь выбрал идентификацию по цвету."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    await q.edit_message_text(t(lang, "ask_shirt"))
    return SHIRT_COLOR


async def identify_position(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь выбрал идентификацию по позиции."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "btn_near"), callback_data="pos_near")],
        [InlineKeyboardButton(t(lang, "btn_far"), callback_data="pos_far")],
    ])
    await q.edit_message_text(t(lang, "ask_position"), reply_markup=kb)
    return POSITION


async def identify_both(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь не знает — анализируем обоих/ближнего."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    context.user_data["identify_mode"] = "both"
    await q.edit_message_text(t(lang, "ask_video"))
    return VIDEO


async def get_shirt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = db.get_lang(uid)
    context.user_data["identify_mode"] = "color"
    context.user_data["shirt"] = update.message.text.strip()[:100]
    await update.message.reply_text(t(lang, "ask_video"))
    return VIDEO


async def get_position(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь выбрал сторону корта."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    context.user_data["identify_mode"] = "position"
    context.user_data["position"] = "near" if q.data == "pos_near" else "far"
    await q.edit_message_text(t(lang, "ask_video"))
    return VIDEO


async def process_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = db.get_lang(uid)
    name = update.effective_user.first_name

    # Строим описание как найти игрока — зависит от выбранного режима
    mode = context.user_data.get("identify_mode", "both")
    if mode == "color":
        target_desc = t(lang, "identify_color", shirt=context.user_data.get("shirt", "?"))
    elif mode == "position":
        pos_key = "pos_near" if context.user_data.get("position") == "near" else "pos_far"
        target_desc = t(lang, "identify_position", position=t(lang, pos_key))
    else:
        target_desc = t(lang, "identify_both")

    if uid in processing_users:
        await update.message.reply_text(t(lang, "busy"))
        return ConversationHandler.END

    processing_users.add(uid)
    msg = await update.message.reply_text(t(lang, "processing"))
    tmpdir = tempfile.mkdtemp()
    loop = asyncio.get_event_loop()

    try:
        video_path = f"{tmpdir}/video.mp4"
        frames_dir = f"{tmpdir}/frames"
        os.makedirs(frames_dir)

        # Скачивание
        if update.message.document or update.message.video:
            fo = update.message.document or update.message.video
            tf = await context.bot.get_file(fo.file_id)
            await tf.download_to_drive(video_path)
        elif update.message.text and "http" in update.message.text:
            await msg.edit_text(t(lang, "downloading"))
            ok = await loop.run_in_executor(
                None, _download_video_sync, update.message.text.strip(), video_path)
            if not ok:
                await msg.edit_text(t(lang, "err_download"))
                return ConversationHandler.END
        else:
            await msg.edit_text(t(lang, "err_format"))
            return ConversationHandler.END

        # Нарезка кадров
        await msg.edit_text(t(lang, "extracting"))
        frames = await loop.run_in_executor(
            None, _extract_frames_sync, video_path, frames_dir)
        if len(frames) < 3:
            logger.warning("Мало кадров (%d) для user %s", len(frames), uid)
            await msg.edit_text(t(lang, "err_frames"))
            return ConversationHandler.END

        # Списываем кредит ТОЛЬКО когда уверены что анализ пойдёт.
        # Админ не платит.
        if not _is_admin(uid):
            if not db.consume_credit(uid):
                await msg.edit_text(t(lang, "no_credits"),
                                    reply_markup=buy_keyboard(lang))
                return ConversationHandler.END

        # AI анализ с таймаутом
        await msg.edit_text(t(lang, "analyzing", n=len(frames)))
        try:
            report = await asyncio.wait_for(
                loop.run_in_executor(None, _analyze_sync,
                                     frames, target_desc, name, lang),
                timeout=ANALYSIS_TIMEOUT)
        except asyncio.TimeoutError:
            if not _is_admin(uid):
                db.add_credits(uid, 1)  # вернуть кредит
            await msg.edit_text(t(lang, "err_timeout"))
            return ConversationHandler.END

        # PDF
        await msg.edit_text(t(lang, "generating"))
        pdf_path = f"{tmpdir}/report.pdf"
        await loop.run_in_executor(
            None, _generate_pdf_sync, report, name, len(frames), lang, pdf_path)

        db.log_analysis(uid, len(frames), shirt)
        remaining = db.get_credits(uid)

        with open(pdf_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=f"RallyIQ_{name}_{datetime.now().strftime('%d%m%Y')}.pdf",
                caption=t(lang, "done", name=name, credits=remaining),
                reply_markup=main_keyboard(lang),
            )
        await msg.delete()

    except Exception as e:
        logger.exception("Ошибка при обработке видео для %s: %s", uid, e)
        await msg.edit_text(t(lang, "err_general"))
    finally:
        processing_users.discard(uid)
        shutil.rmtree(tmpdir, ignore_errors=True)

    return ConversationHandler.END


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = db.get_lang(update.effective_user.id)
    await update.message.reply_text(
        t(lang, "cancelled"), reply_markup=main_keyboard(lang))
    return ConversationHandler.END



# ==================================================
# АДМИН-КОМАНДЫ (только для ADMIN_ID)
# ==================================================
async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главная админ-панель."""
    if not _is_admin(update.effective_user.id):
        return
    text = (
        "🛠 *Админ-панель RallyIQ*\n\n"
        "/stats — статистика проекта\n"
        "/users — последние пользователи\n"
        "/give <user_id> <кол-во> — начислить кредиты\n"
        "/giveme <кол-во> — начислить себе\n\n"
        "Ты администратор — все анализы бесплатны."
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def admin_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_user.id):
        return
    s = db.get_stats()
    text = (
        "📊 *Статистика RallyIQ*\n\n"
        f"👥 Всего пользователей: {s['total_users']}\n"
        f"🎬 Всего анализов: {s['total_analyses']}\n"
        f"💳 Кредитов на балансах: {s['total_credits']}\n"
        f"🔥 Анализов за 7 дней: {s['active_week']}"
    )
    await update.message.reply_text(text, parse_mode="Markdown")


async def admin_users(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _is_admin(update.effective_user.id):
        return
    users = db.get_recent_users(15)
    if not users:
        await update.message.reply_text("Пока нет пользователей.")
        return
    lines = ["👥 *Последние пользователи:*\n"]
    for u in users:
        uname = f"@{u['username']}" if u["username"] else "—"
        lines.append(
            f"`{u['user_id']}` {u['first_name'] or ''} {uname}\n"
            f"   💳 {u['credits']} | 🎬 {u['analyses_count']} | 🌐 {u['lang']}"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def admin_give(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/give <user_id> <amount> — начислить кредиты пользователю."""
    if not _is_admin(update.effective_user.id):
        return
    args = context.args
    if len(args) != 2:
        await update.message.reply_text("Использование: /give <user_id> <кол-во>")
        return
    try:
        target_id = int(args[0])
        amount = int(args[1])
    except ValueError:
        await update.message.reply_text("user_id и кол-во должны быть числами.")
        return
    new_balance = db.add_credits_by_id(target_id, amount)
    if new_balance is None:
        await update.message.reply_text(
            f"❌ Пользователь {target_id} не найден.\n"
            "Он должен сначала написать /start боту."
        )
        return
    await update.message.reply_text(
        f"✅ Начислено {amount} анализов пользователю {target_id}.\n"
        f"Новый баланс: {new_balance}"
    )
    # Уведомляем пользователя
    try:
        await context.bot.send_message(
            target_id,
            f"🎁 Тебе начислено {amount} анализов! Баланс: {new_balance}"
        )
    except Exception:
        pass


async def admin_giveme(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/giveme <amount> — начислить себе."""
    if not _is_admin(update.effective_user.id):
        return
    args = context.args
    amount = int(args[0]) if args and args[0].isdigit() else 10
    db.ensure_user(update.effective_user.id, update.effective_user.username,
                   update.effective_user.first_name)
    new_balance = db.add_credits_by_id(update.effective_user.id, amount)
    await update.message.reply_text(f"✅ Начислено {amount}. Баланс: {new_balance}")


async def noop_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Заглушка для неактивных кнопок."""
    await update.callback_query.answer("Скоро будет доступно", show_alert=False)


# ==================================================
# ЗАПУСК
# ==================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Логирует все необработанные ошибки вместо краша."""
    logger.error("Необработанная ошибка: %s", context.error, exc_info=context.error)


async def post_init(app: Application) -> None:
    await app.bot.set_my_commands([
        ("analyze", "🏸 Analyze video"),
        ("buy", "💳 Buy analyses"),
        ("free", "🎁 Free analysis"),
        ("balance", "💰 My balance"),
        ("help", "❓ Help"),
    ])
    logger.info("Команды меню установлены")


def main() -> None:
    db.init_pool()
    db.init_schema()

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("analyze", analyze_entry_cmd),
            CallbackQueryHandler(analyze_entry_btn, pattern="^go_analyze$"),
        ],
        states={
            IDENTIFY: [
                CallbackQueryHandler(identify_color, pattern="^id_color$"),
                CallbackQueryHandler(identify_position, pattern="^id_position$"),
                CallbackQueryHandler(identify_both, pattern="^id_both$"),
            ],
            SHIRT_COLOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_shirt)],
            POSITION: [
                CallbackQueryHandler(get_position, pattern="^pos_(near|far)$"),
            ],
            VIDEO: [
                MessageHandler(filters.Document.ALL | filters.VIDEO, process_video),
                MessageHandler(filters.TEXT & ~filters.COMMAND, process_video),
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(CommandHandler("free", free_analysis))
    app.add_handler(CommandHandler("balance", balance))
    # Админские команды
    app.add_handler(CommandHandler("admin", admin_panel))
    app.add_handler(CommandHandler("stats", admin_stats))
    app.add_handler(CommandHandler("users", admin_users))
    app.add_handler(CommandHandler("give", admin_give))
    app.add_handler(CommandHandler("giveme", admin_giveme))
    app.add_handler(conv)
    app.add_handler(CallbackQueryHandler(lang_callback, pattern="^lang_"))
    app.add_handler(CallbackQueryHandler(buy_callback, pattern="^go_buy$"))
    app.add_handler(CallbackQueryHandler(paid_callback, pattern="^paid_5$"))
    app.add_handler(CallbackQueryHandler(noop_callback, pattern="^noop$"))

    app.add_error_handler(error_handler)
    logger.info("RallyIQ запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
