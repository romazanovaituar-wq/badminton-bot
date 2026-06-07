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
                      ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove,
                      LabeledPrice)
from telegram.ext import (Application, CommandHandler, MessageHandler,
                          CallbackQueryHandler, ContextTypes,
                          filters, ConversationHandler, PreCheckoutQueryHandler)
from openai import OpenAI
from fpdf import FPDF
import cv2

import db
import pose
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
# Пакеты оплаты в Telegram Stars (XTR): payload -> (звёзды, кредиты, название)
STAR_PACKAGES = {
    "pack_1":  {"stars": 75,   "credits": 1,  "title": "1 анализ"},
    "pack_5":  {"stars": 300,  "credits": 5,  "title": "5 анализов"},
    "pack_20": {"stars": 1000, "credits": 20, "title": "20 анализов"},
}

# Реквизиты для ручной оплаты (Казахстан, Kaspi).
# ВПИШИ свой номер Kaspi вместо заглушки ниже.
KASPI_NUMBER = os.environ.get("KASPI_NUMBER", "+7 XXX XXX XX XX")
KASPI_NAME = os.environ.get("KASPI_NAME", "Aituar")
# Цены в тенге за пакеты (примерно $1/анализ при курсе ~480₸)
KZT_PRICES = {
    "1 анализ": "500₸",
    "5 анализов": "2000₸",
    "20 анализов": "7000₸",
}

MOTION_THRESHOLD = 12
MAX_FRAMES       = 25
ANALYSIS_TIMEOUT = 600  # 10 минут максимум на анализ
MAX_VIDEO_MB     = 200
ADMIN_ID         = 942577691  # Telegram ID администратора
ADMIN_USERNAME   = os.environ.get("ADMIN_USERNAME", "@N1world1N")  # для связи по оплате


def _is_admin(user_id: int) -> bool:
    return user_id == ADMIN_ID

GAME_TYPE, IDENTIFY, SHIRT_COLOR, POSITION, VIDEO = range(5)

client = OpenAI(api_key=OPENAI_API_KEY, timeout=60.0, max_retries=2)

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

def menu_keyboard(lang: str) -> ReplyKeyboardMarkup:
    """Постоянное меню внизу экрана — всегда видно, не нужно помнить команды."""
    return ReplyKeyboardMarkup(
        [
            [KeyboardButton(t(lang, "menu_analyze")), KeyboardButton(t(lang, "menu_buy"))],
            [KeyboardButton(t(lang, "menu_balance")), KeyboardButton(t(lang, "menu_language"))],
            [KeyboardButton(t(lang, "menu_faq"))],
        ],
        resize_keyboard=True,
        is_persistent=True,
    )


def main_keyboard(lang: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "btn_analyze"), callback_data="go_analyze")],
        [InlineKeyboardButton(t(lang, "btn_buy"), callback_data="go_buy")],
    ])


def buy_keyboard(lang: str) -> InlineKeyboardMarkup:
    """Кнопки выбора пакета — каждая запускает выставление счёта в Stars."""
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(
            f"⭐ {STAR_PACKAGES['pack_1']['stars']} — {t(lang, 'pack_1')}",
            callback_data="buy_pack_1")],
        [InlineKeyboardButton(
            f"⭐ {STAR_PACKAGES['pack_5']['stars']} — {t(lang, 'pack_5')}",
            callback_data="buy_pack_5")],
        [InlineKeyboardButton(
            f"⭐ {STAR_PACKAGES['pack_20']['stars']} — {t(lang, 'pack_20')}",
            callback_data="buy_pack_20")],
        [InlineKeyboardButton(
            t(lang, "btn_kaspi"), callback_data="pay_kaspi")],
    ])


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
        # Гибкий выбор формата с фоллбэками — берём что доступно
        "format": "mp4/bestvideo[height<=720]+bestaudio/best",
        "outtmpl": dest_path,
        "noplaylist": True,
        "max_filesize": MAX_VIDEO_MB * 1024 * 1024,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        # Несколько client-ов для обхода блокировок YouTube
        "extractor_args": {
            "youtube": {"player_client": ["android", "ios", "web", "tv"]}
        },
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error("yt-dlp ошибка: %s", e)
        return False
    return os.path.exists(dest_path)


def _download_gdrive(url: str, dest_path: str) -> bool:
    """
    Скачивает файл с Google Drive через библиотеку gdown,
    которая корректно обходит страницу подтверждения для больших файлов.
    """
    import re
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if not m:
        logger.error("Не удалось извлечь file_id из ссылки Google Drive")
        return False
    file_id = m.group(1)
    try:
        import gdown
        gdown.download(id=file_id, output=dest_path, quiet=True, fuzzy=True)
    except Exception as e:
        logger.error("gdown ошибка: %s", e)
        return False
    ok = os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000
    if not ok:
        logger.error("Google Drive: файл не скачался или слишком мал. "
                     "Проверь что доступ открыт 'всем у кого есть ссылка'.")
    return ok

def _save_frame(frame, output_dir: str, index: int, time_s: float,
                rotation: int = 0) -> dict:
    """Сохраняет кадр с ресайзом + коррекцией ориентации."""
    # Поворот по метаданным видео (надёжно). Спорные случаи (нет метаданных,
    # но кадр вертикальный) решаются ВОПРОСОМ пользователю до анализа —
    # см. логику в process_video, здесь только применяем известный угол.
    if rotation:
        frame = _auto_rotate_frame(frame, rotation)
    h, w = frame.shape[:2]
    if w > 800:
        frame = cv2.resize(frame, (800, int(h * 800 / w)))
    path = f"{output_dir}/frame_{index:03d}.jpg"
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"path": path, "time": time_s}


def _get_video_rotation(video_path: str) -> int:
    """
    Читает угол ротации видео из метаданных (EXIF/MP4).
    Вертикальное видео с телефона обычно имеет rotation=90 или 270.
    Возвращает угол (0, 90, 180, 270) или 0 если не определить.
    """
    try:
        import subprocess, json
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", video_path],
            capture_output=True, text=True, timeout=10
        )
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            # Ротация бывает в tags или side_data_list
            tags = stream.get("tags", {})
            rot = tags.get("rotate") or tags.get("Rotate")
            if rot:
                return int(rot)
            for sd in stream.get("side_data_list", []):
                if sd.get("side_data_type") == "Display Matrix":
                    rot = sd.get("rotation")
                    if rot is not None:
                        return abs(int(rot))
    except Exception:
        pass
    return 0


def _auto_rotate_frame(frame, rotation: int):
    """
    Поворачивает кадр на нужный угол чтобы компенсировать ротацию видео.
    rotation — угол из метаданных видео.
    """
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame  # 0 или неизвестный — не трогаем


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
    # Определяем ротацию видео (телефоны часто снимают вертикально)
    # Поворот по метаданным видео (если есть тег rotate)
    rotation = _get_video_rotation(video_path)
    if rotation:
        logger.info("Видео ротация из метаданных: %d°", rotation)

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
                    motion.append(_save_frame(frame, output_dir, len(motion), idx / fps, rotation))
                    last_saved = idx
            prev_gray = gray
        idx += 1

    if len(motion) >= 6:
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

    # Берём кадров больше чем нужно, потом оставим самые "динамичные"
    uniform: list[dict] = []
    candidates: list[tuple] = []  # (frame, time, motion_score)
    prev_small = None
    if total > 0:
        # Шаг мельче — набираем больше кандидатов для отбора по движению
        step = max(1, total // (MAX_FRAMES * 2))
        for i in range(0, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            # Оценка движения относительно предыдущего кадра
            small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            score = 0.0
            if prev_small is not None:
                score = cv2.absdiff(small, prev_small).mean()
            prev_small = small
            candidates.append((frame, i / fps, score))

    if candidates:
        # Сортируем по убыванию движения, берём топ MAX_FRAMES самых активных,
        # затем восстанавливаем хронологический порядок по времени.
        # Так в анализ попадают игровые моменты, а не паузы/отдых.
        active = sorted(candidates, key=lambda c: c[2], reverse=True)[:MAX_FRAMES]
        active.sort(key=lambda c: c[1])  # по времени
        for frame, t_sec, _score in active:
            uniform.append(_save_frame(frame, output_dir, len(uniform), t_sec, rotation))

    if len(uniform) >= 3:
        cap.release()
        logger.info("Извлечено %d кадров (uniform+motion-фильтр)", len(uniform))
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
            any_frames.append(_save_frame(frame, output_dir, len(any_frames), idx / fps, rotation))
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


def _validate_badminton(frames: list[dict]) -> bool:
    """
    Проверяет что на видео действительно бадминтон, ДО полного анализа.
    Смотрит 3 кадра из разных частей видео. Дёшево (gpt-4o-mini).
    Защищает от мусорных видео и слива денег на галлюцинации.
    """
    # Берём до 3 кадров: начало, середина, конец
    sample = []
    if len(frames) >= 3:
        sample = [frames[0], frames[len(frames)//2], frames[-1]]
    else:
        sample = frames

    content = []
    for frame in sample:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{_encode_image(frame['path'])}",
                "detail": "low",
            },
        })
    content.append({
        "type": "text",
        "text": (
            "Look at these frames. Is this a BADMINTON match or training "
            "(court, players with rackets, shuttlecock, badminton movements)? "
            "Answer with ONLY one word: YES if it is badminton, NO if it is "
            "anything else (street, people walking, random objects, other sports, memes, etc.)."
        ),
    })

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=5,
            messages=[{"role": "user", "content": content}],
        )
        answer = resp.choices[0].message.content.strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        logger.error("Ошибка валидации бадминтона: %s", e)
        # При ошибке — пропускаем (не блокируем пользователя из-за сбоя)
        return True


def _analyze_sync(frames: list[dict], target: str,
                  name: str, lang: str) -> str:
    frame_prompt = t(lang, "frame_prompt", target=target)
    descriptions: list[str] = []
    not_found_count = 0

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
        desc = resp.choices[0].message.content or ""
        up = desc.upper()
        if "TARGET_NOT_FOUND" in up:
            not_found_count += 1
        elif "NOT VISIBLE" in up:
            pass  # в этом кадре игрока не видно — пропускаем
        else:
            # Всё остальное (включая PAUSE-кадры) идёт в анализ.
            # GPT сам учтёт что некоторые моменты — паузы.
            descriptions.append(f"Frame {i+1} ({frame['time']:.0f}s): {desc}")

    # Отказываем ТОЛЬКО если игрок реально отсутствует на видео:
    # GPT явно сказал "не тот игрок" в большинстве кадров.
    # PAUSE (отдых) и NOT VISIBLE НЕ считаются за отсутствие игрока —
    # это нормальные моменты, просто их не оцениваем как технику.
    total = len(frames)
    if total > 0 and not_found_count >= total * 0.7:
        return "TARGET_NOT_FOUND"

    # Отказ только если ВООБЩЕ ничего не описано (0 кадров с игроком).
    # Если есть хотя бы 1 описание — анализируем (частичный анализ лучше отказа).
    if len(descriptions) == 0:
        return "NOT_ENOUGH_DATA"

    # РЕАЛЬНЫЕ ИЗМЕРЕНИЯ через MediaPipe (если доступен).
    # Считаем метрики позы по кадрам и даём GPT как объективные данные,
    # чтобы оценка опиралась на измерения, а не только на догадки по картинкам.
    pose_note = ""
    measured_scores = None
    try:
        frame_paths = [f["path"] for f in frames]
        metrics = pose.analyze_frames(frame_paths)
        if metrics:
            summary = pose.metrics_summary(metrics, lang)
            measured_scores = pose.compute_scores(metrics)
            if summary:
                logger.info("Pose-метрики: %s", summary)
                level_hint = ""
                if measured_scores:
                    logger.info("Измеренные баллы: %s", measured_scores)
                    lvl = pose.level_from_scores(measured_scores)
                    if lvl:
                        level_hint = (
                            f"\n\nПо измеренным данным уровень игрока: {lvl}. "
                            f"НЕ занижай уровень — если измерения показывают высокий "
                            f"класс, так и пиши. Не называй всех любителями."
                        )
                pose_note = (
                    "\n\nОБЪЕКТИВНЫЕ ИЗМЕРЕНИЯ позы игрока (данные компьютерного "
                    "зрения, опирайся на них при оценке техники и работы ног): "
                    + summary + level_hint
                )
    except Exception as e:
        logger.warning("Pose-анализ пропущен: %s", e)

    report_prompt = t(lang, "report_prompt",
                      n=len(descriptions), name=name, target=target,
                      frames="\n".join(descriptions)) + pose_note
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=[{"role": "user", "content": report_prompt}],
    )
    report = resp.choices[0].message.content or ""

    # Переопределяем баллы ИЗМЕРЕННЫМИ значениями там где они есть.
    # Это делает footwork/technique/positioning/recovery воспроизводимыми.
    if measured_scores:
        report = _apply_measured_scores(report, measured_scores)

    return report


def _apply_measured_scores(report: str, measured: dict) -> str:
    """
    Заменяет в строке SCORES баллы на измеренные (footwork/technique/
    positioning/recovery). Tactics и overall оставляем от GPT (их не измеришь).
    Формат строки: SCORES: overall|footwork|technique|tactics|positioning|recovery
    """
    import re
    lines = report.split("\n")
    for idx, line in enumerate(lines):
        mt = re.match(r"\s*SCORES:\s*(.+)", line, re.IGNORECASE)
        if not mt:
            continue
        parts = [p.strip() for p in mt.group(1).split("|")]
        if len(parts) < 6:
            break
        try:
            nums = [int(re.sub(r"[^0-9]", "", p) or 0) for p in parts]
        except ValueError:
            break
        # nums: [overall, footwork, technique, tactics, positioning, recovery]
        if "footwork" in measured:
            nums[1] = measured["footwork"]
        if "technique" in measured:
            nums[2] = measured["technique"]
        if "positioning" in measured:
            nums[4] = measured["positioning"]
        if "recovery" in measured:
            nums[5] = measured["recovery"]
        # overall пересчитываем как среднее всех пяти
        nums[0] = int(round((nums[1] + nums[2] + nums[3] + nums[4] + nums[5]) / 5))
        lines[idx] = "SCORES: " + "|".join(str(n) for n in nums)
        break
    return "\n".join(lines)


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


def _section_color(title: str) -> tuple:
    """Возвращает цвет акцента для секции по ключевым словам (RU/KZ/EN)."""
    t_low = title.lower()
    # Приоритет №1 — красный (привлекает внимание к главному)
    if any(w in t_low for w in ["приоритет", "басымдық", "priority"]):
        return (255, 90, 90)
    # Сильные стороны — бирюзовый/зелёный
    if any(w in t_low for w in ["сильн", "күшті", "strength"]):
        return (0, 200, 150)
    # Ошибки — оранжевый
    if any(w in t_low for w in ["ошибк", "қател", "mistake"]):
        return (255, 140, 60)
    # Тактика — голубой
    if any(w in t_low for w in ["тактик", "tactic"]):
        return (80, 170, 255)
    # Твой уровень — светло-голубой
    if any(w in t_low for w in ["уровень", "деңгей", "level"]):
        return (120, 200, 255)
    # Упражнения — фиолетовый
    if any(w in t_low for w in ["упражнен", "жаттығу", "drill"]):
        return (180, 130, 255)
    # План на неделю — золотой
    if any(w in t_low for w in ["план", "жоспар", "plan"]):
        return (240, 190, 90)
    # Итог — золотой
    if any(w in t_low for w in ["итог", "қорытынд", "summary"]):
        return (240, 190, 90)
    return (120, 200, 255)


def _parse_scores(report: str) -> tuple:
    """
    Извлекает строку SCORES из отчёта GPT.
    Возвращает (scores_dict | None, report_без_строки_scores).
    """
    import re
    scores = None
    lines = report.split("\n")
    clean_lines = []
    for line in lines:
        m = re.match(r"\s*SCORES:\s*(.+)", line, re.IGNORECASE)
        if m and scores is None:
            parts = [p.strip() for p in m.group(1).split("|")]
            nums = []
            for p in parts:
                try:
                    nums.append(max(0, min(100, int(re.sub(r"[^0-9]", "", p)))))
                except (ValueError, TypeError):
                    nums.append(0)
            if len(nums) >= 6:
                scores = {
                    "overall": nums[0],
                    "footwork": nums[1],
                    "technique": nums[2],
                    "tactics": nums[3],
                    "positioning": nums[4],
                    "recovery": nums[5],
                }
            continue  # строку SCORES не добавляем в текст
        clean_lines.append(line)
    return scores, "\n".join(clean_lines)


def _score_color(value: int) -> tuple:
    """Цвет полоски по баллу: красный/жёлтый/зелёный."""
    if value < 60:
        return (230, 90, 80)    # красный
    if value < 75:
        return (240, 190, 90)   # жёлтый/золотой
    return (0, 200, 150)        # зелёный


def _generate_pdf_sync(report: str, name: str, frames_count: int,
                       lang: str, out_path: str,
                       skeleton_path: str = None,
                       heatmap_path: str = None,
                       extra_metrics: dict = None) -> str:
    _ensure_fonts()

    # Извлекаем оценки игрока (если GPT их выдал)
    scores, report = _parse_scores(report)

    # Премиум тёмная палитра
    BG_DARK   = (18, 22, 33)      # фон страницы
    CARD      = (28, 34, 49)      # карточки секций
    GOLD      = (240, 190, 90)    # золотой акцент
    WHITE     = (235, 240, 248)
    GREY      = (150, 160, 178)

    class Report(FPDF):
        def header(self):
            # Тёмный фон всей страницы
            self.set_fill_color(*BG_DARK)
            self.rect(0, 0, self.w, self.h, "F")

        def footer(self):
            self.set_y(-14)
            self.set_font("Roboto", "", 8)
            self.set_text_color(*GREY)
            self.cell(0, 8,
                      f"RallyIQ  ·  {datetime.now().strftime('%d.%m.%Y')}  ·  {self.page_no()}",
                      align="C")

    pdf = Report()
    pdf.add_font("Roboto", "", FONT_REG)
    pdf.add_font("Roboto", "B", FONT_BLD)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ====== ОБЛОЖКА-ШАПКА ======
    # Золотая полоса сверху
    pdf.set_fill_color(*GOLD)
    pdf.rect(0, 0, pdf.w, 3, "F")

    pdf.set_y(18)
    pdf.set_font("Roboto", "B", 26)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 14, "RallyIQ", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Roboto", "", 12)
    pdf.set_text_color(*GOLD)
    pdf.cell(0, 7, "AI BADMINTON COACH", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Карточка с инфо об игроке
    pdf.set_fill_color(*CARD)
    card_y = pdf.get_y()
    pdf.rect(pdf.l_margin, card_y, pdf.w - 2*pdf.l_margin, 22, "F")
    # Золотая вертикальная полоска слева карточки
    pdf.set_fill_color(*GOLD)
    pdf.rect(pdf.l_margin, card_y, 1.5, 22, "F")

    pdf.set_xy(pdf.l_margin + 6, card_y + 4)
    pdf.set_font("Roboto", "B", 14)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 7, name, new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(pdf.l_margin + 6)
    pdf.set_font("Roboto", "", 9)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 6,
             f"{datetime.now().strftime('%d.%m.%Y')}   ·   проанализировано кадров: {frames_count}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ====== СКЕЛЕТ (визуальное доказательство CV) ======
    if skeleton_path and os.path.exists(skeleton_path):
        try:
            if pdf.get_y() > pdf.h - 110:
                pdf.add_page()
            cap = {"ru": "AI-анализ позы (компьютерное зрение)",
                   "kz": "AI-поза талдауы (компьютерлік көру)",
                   "en": "AI pose analysis (computer vision)"}.get(lang, "")
            pdf.set_font("Roboto", "B", 11)
            pdf.set_text_color(*GOLD)
            pdf.cell(0, 7, cap, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            # Вставляем картинку по центру, ширина 85мм. Высоту считаем
            # по реальным пропорциям картинки, чтобы курсор не наложился.
            img_w = 85
            try:
                import cv2 as _cv2
                _im = _cv2.imread(skeleton_path)
                ratio = (_im.shape[0] / _im.shape[1]) if _im is not None else 0.6
            except Exception:
                ratio = 0.6
            img_h = img_w * ratio
            x = (pdf.w - img_w) / 2
            y = pdf.get_y()
            pdf.image(skeleton_path, x=x, y=y, w=img_w)
            pdf.set_y(y + img_h + 8)  # реальная высота + отступ
        except Exception:
            pass

    # ====== ТЕПЛОВАЯ КАРТА ПЕРЕМЕЩЕНИЙ ======
    if heatmap_path and os.path.exists(heatmap_path):
        try:
            # Карта высокая (~80мм). Если мало места — новая страница.
            if pdf.get_y() > pdf.h - 100:
                pdf.add_page()
            cap = {"ru": "Карта перемещений по корту",
                   "kz": "Корт бойынша қозғалыс картасы",
                   "en": "Court movement heatmap"}.get(lang, "")
            pdf.set_font("Roboto", "B", 11)
            pdf.set_text_color(*GOLD)
            pdf.cell(0, 7, cap, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            img_w = 60
            try:
                import cv2 as _cv2
                _im = _cv2.imread(heatmap_path)
                ratio = (_im.shape[0] / _im.shape[1]) if _im is not None else 1.33
            except Exception:
                ratio = 1.33
            x = (pdf.w - img_w) / 2
            y = pdf.get_y()
            pdf.image(heatmap_path, x=x, y=y, w=img_w)
            pdf.set_y(y + img_w * ratio + 8)
        except Exception:
            pass

    # ====== БЛОК ИЗМЕРЕННЫХ МЕТРИК ======
    if extra_metrics:
        try:
            labels = {
                "ru": {"title": "ИЗМЕРЕНИЯ (компьютерное зрение)",
                       "dist": "Пройдено на корте", "stretch": "Растяжка выпадов",
                       "knee": "Сгиб колен", "active": "Активность"},
                "kz": {"title": "ӨЛШЕМДЕР (компьютерлік көру)",
                       "dist": "Кортта жүгірді", "stretch": "Шалқу тереңдігі",
                       "knee": "Тізе бүгу", "active": "Белсенділік"},
                "en": {"title": "MEASUREMENTS (computer vision)",
                       "dist": "Court distance", "stretch": "Lunge stretch",
                       "knee": "Knee flex", "active": "Activity"},
            }.get(lang, None)
            if labels:
                pdf.set_font("Roboto", "B", 11)
                pdf.set_text_color(*GOLD)
                pdf.cell(0, 7, labels["title"], new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)
                pdf.set_font("Roboto", "", 10)
                pdf.set_text_color(220, 220, 220)
                knee = extra_metrics.get("knee_flex")
                ls = extra_metrics.get("leg_stretch")
                mv = extra_metrics.get("movement")
                # Словесные оценки вместо сырых чисел — понятнее игроку
                def word_level(val, lo, hi, lang):
                    w = {"ru": ["низкая", "средняя", "высокая"],
                         "kz": ["төмен", "орташа", "жоғары"],
                         "en": ["low", "medium", "high"]}.get(lang, ["low","mid","high"])
                    if val < lo: return w[0]
                    if val < hi: return w[1]
                    return w[2]
                rows = []
                if knee is not None:
                    rows.append(f"{labels['knee']}: {knee}°")
                if ls is not None:
                    rows.append(f"{labels['stretch']}: {round(ls*100)}%")
                if mv is not None:
                    rows.append(f"{labels['active']}: {word_level(mv, 2.5, 6, lang)}")
                for r in rows:
                    pdf.cell(0, 6, "  • " + r, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)
        except Exception:
            pass

    # ====== БЛОК ОЦЕНОК ======
    if scores:
        # Если до конца страницы мало места — начинаем блок с новой страницы,
        # чтобы оценки не разорвались между страницами.
        if pdf.get_y() > pdf.h - 90:
            pdf.add_page()
        labels = {
            "ru": {"title": "ОЦЕНКА ИГРЫ", "footwork": "Работа ног",
                   "technique": "Техника", "tactics": "Тактика",
                   "positioning": "Позиционирование", "recovery": "Восстановление"},
            "kz": {"title": "ОЙЫН БАҒАСЫ", "footwork": "Аяқ жұмысы",
                   "technique": "Техника", "tactics": "Тактика",
                   "positioning": "Позиция", "recovery": "Қалпына келу"},
            "en": {"title": "PERFORMANCE SCORE", "footwork": "Footwork",
                   "technique": "Technique", "tactics": "Tactics",
                   "positioning": "Positioning", "recovery": "Recovery"},
        }.get(lang, None)
        if labels is None:
            labels = {"title": "PERFORMANCE SCORE", "footwork": "Footwork",
                      "technique": "Technique", "tactics": "Tactics",
                      "positioning": "Positioning", "recovery": "Recovery"}

        # Общий балл — крупно
        overall = scores["overall"]
        ov_color = _score_color(overall)
        pdf.set_font("Roboto", "B", 11)
        pdf.set_text_color(*GOLD)
        pdf.cell(0, 7, labels["title"], new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

        y0 = pdf.get_y()
        pdf.set_font("Roboto", "B", 40)
        pdf.set_text_color(*ov_color)
        pdf.cell(40, 18, str(overall), new_x="RIGHT", new_y="TOP")
        pdf.set_font("Roboto", "", 12)
        pdf.set_text_color(*GREY)
        pdf.set_xy(pdf.l_margin + 32, y0 + 9)
        pdf.cell(0, 8, "/ 100", new_x="LMARGIN", new_y="NEXT")
        pdf.set_y(y0 + 20)
        pdf.ln(2)

        # Полоски по категориям
        cats = ["footwork", "technique", "tactics", "positioning", "recovery"]
        bar_x = pdf.l_margin + 48
        bar_w = pdf.w - pdf.l_margin - bar_x - 12
        for cat in cats:
            val = scores[cat]
            col = _score_color(val)
            yc = pdf.get_y()
            # Подпись
            pdf.set_font("Roboto", "", 9)
            pdf.set_text_color(*WHITE)
            pdf.set_xy(pdf.l_margin, yc)
            pdf.cell(46, 6, labels[cat], new_x="RIGHT", new_y="TOP")
            # Фон полоски
            pdf.set_fill_color(40, 48, 66)
            pdf.rect(bar_x, yc + 1.5, bar_w, 3.5, "F")
            # Заполнение
            pdf.set_fill_color(*col)
            pdf.rect(bar_x, yc + 1.5, bar_w * val / 100, 3.5, "F")
            # Число
            pdf.set_font("Roboto", "B", 9)
            pdf.set_text_color(*col)
            pdf.set_xy(pdf.w - pdf.l_margin - 12, yc)
            pdf.cell(12, 6, str(val), align="R", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        pdf.ln(6)

    # ====== СЕКЦИИ ======
    current = None
    buffer: list[str] = []

    def flush():
        if not (current and buffer):
            return
        accent = _section_color(current)
        # Заголовок секции с цветной полоской
        pdf.ln(2)
        y = pdf.get_y()
        pdf.set_fill_color(*accent)
        pdf.rect(pdf.l_margin, y + 1, 4, 7, "F")
        pdf.set_x(pdf.l_margin + 7)
        pdf.set_font("Roboto", "B", 13)
        pdf.set_text_color(*accent)
        pdf.cell(0, 9, current.upper(), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
        # Пункты
        pdf.set_text_color(*WHITE)
        for ln in buffer:
            pdf.set_font("Roboto", "", 10)
            pdf.set_x(pdf.l_margin + 2)
            pdf.multi_cell(pdf.w - 2*pdf.l_margin - 4, 6, ln,
                           new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.5)
        pdf.ln(3)

    for line in report.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("## "):
            flush()
            current = line[3:].strip()
            buffer = []
        elif line.startswith("#"):
            continue
        else:
            buffer.append(line)
    flush()

    # ====== ДИСКЛЕЙМЕР ======
    pdf.ln(4)
    pdf.set_draw_color(*GREY)
    pdf.set_line_width(0.2)
    y = pdf.get_y()
    pdf.line(pdf.l_margin, y, pdf.w - pdf.l_margin, y)
    pdf.ln(3)
    pdf.set_font("Roboto", "", 8)
    pdf.set_text_color(*GREY)
    disclaimer = {
        "ru": "Отчёт сгенерирован AI на основе кадров видео и может содержать неточности. Не заменяет очного тренера.",
        "kz": "Есеп бейне кадрлары негізінде AI арқылы жасалған, дәл болмауы мүмкін. Жаттықтырушыны алмастырмайды.",
        "en": "This report is AI-generated from video frames and may contain inaccuracies. Not a substitute for a real coach.",
    }.get(lang, "")
    pdf.multi_cell(pdf.w - 2*pdf.l_margin, 4, disclaimer,
                   new_x="LMARGIN", new_y="NEXT")

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


async def see_demo_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает новичку как работает бот (трейлер по шагам) + пример отчёта."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    # Трейлер: шаги работы
    await q.message.reply_text(t(lang, "demo_steps"))
    # Пример отчёта — что человек получит
    await q.message.reply_text(t(lang, "demo_report"),
                               reply_markup=menu_keyboard(lang))


async def lang_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    lang = q.data.replace("lang_", "")
    db.ensure_user(q.from_user.id, q.from_user.username, q.from_user.first_name)
    db.set_lang(q.from_user.id, lang)
    credits = db.get_credits(q.from_user.id)
    await q.edit_message_text(
        t(lang, "welcome", name=q.from_user.first_name, credits=credits))
    # Онбординг с кнопкой "пример отчёта" — новичок сразу видит что получит
    demo_kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "btn_see_demo"), callback_data="see_demo")],
    ])
    await q.message.reply_text(t(lang, "onboarding"), reply_markup=demo_kb)
    await q.message.reply_text(t(lang, "menu_hint"), reply_markup=menu_keyboard(lang))


async def language_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Позволяет сменить язык в любой момент."""
    u = update.effective_user
    db.ensure_user(u.id, u.username, u.first_name)
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🇷🇺 Русский", callback_data="lang_ru")],
        [InlineKeyboardButton("🇰🇿 Қазақша", callback_data="lang_kz")],
        [InlineKeyboardButton("🇬🇧 English", callback_data="lang_en")],
    ])
    await update.message.reply_text(TEXTS["ru"]["choose_lang"], reply_markup=keyboard)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = db.get_lang(update.effective_user.id)
    await update.message.reply_text(t(lang, "help"))


async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = db.get_lang(uid)
    await update.message.reply_text(
        t(lang, "balance", credits=db.get_credits(uid)),
        reply_markup=menu_keyboard(lang))


async def buy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = db.get_lang(update.effective_user.id)
    await update.message.reply_text(
        t(lang, "buy_text"), reply_markup=buy_keyboard(lang))


async def buy_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    await q.message.reply_text(t(lang, "buy_text"), reply_markup=buy_keyboard(lang))


async def kaspi_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает реквизиты Kaspi, цены в тенге и ID пользователя для ручной оплаты."""
    q = update.callback_query
    await q.answer()
    uid = q.from_user.id
    lang = db.get_lang(uid)
    prices_text = "\n".join(
        f"   {name} — {price}" for name, price in KZT_PRICES.items()
    )
    text = t(lang, "kaspi_info",
             prices=prices_text,
             kaspi=KASPI_NUMBER,
             name=KASPI_NAME,
             uid=uid,
             admin=ADMIN_USERNAME)
    await q.message.reply_text(text)


async def send_invoice_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь выбрал пакет — выставляем счёт в Telegram Stars."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    payload = q.data.replace("buy_", "")  # pack_1 / pack_5 / pack_20
    pack = STAR_PACKAGES.get(payload)
    if not pack:
        return
    # Счёт в Stars: provider_token пустой, валюта XTR
    await context.bot.send_invoice(
        chat_id=q.from_user.id,
        title=t(lang, "invoice_title", credits=pack["credits"]),
        description=t(lang, "invoice_desc", credits=pack["credits"]),
        payload=payload,
        provider_token="",          # для Stars — пусто
        currency="XTR",
        prices=[LabeledPrice(t(lang, "invoice_label"), pack["stars"])],
    )


async def precheckout_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Подтверждаем счёт перед оплатой (обязательный шаг Telegram)."""
    query = update.pre_checkout_query
    if query.invoice_payload in STAR_PACKAGES:
        await query.answer(ok=True)
    else:
        await query.answer(ok=False, error_message="Unknown package")


async def successful_payment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Платёж прошёл — начисляем кредиты и логируем."""
    uid = update.effective_user.id
    lang = db.get_lang(uid)
    payment = update.message.successful_payment
    payload = payment.invoice_payload
    pack = STAR_PACKAGES.get(payload)
    if not pack:
        return
    new_balance = db.add_credits(uid, pack["credits"])
    db.log_payment(uid, pack["stars"], pack["credits"],
                   payment.telegram_payment_charge_id)
    await update.message.reply_text(
        t(lang, "payment_ok", credits=pack["credits"], balance=new_balance),
        reply_markup=menu_keyboard(lang),
    )
    logger.info("Оплата: user %s, %d звёзд, +%d кредитов",
                uid, pack["stars"], pack["credits"])


async def free_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = db.get_lang(uid)
    db.ensure_user(uid, update.effective_user.username, update.effective_user.first_name)
    if db.grant_free(uid):
        await update.message.reply_text(
            t(lang, "free_ok"), reply_markup=menu_keyboard(lang))
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

    # Режим обслуживания: не запускаем анализ, вежливо предупреждаем.
    # Админ работает всегда (чтобы тестировать).
    if not _is_admin(uid) and db.get_setting("maintenance", "0") == "1":
        await message.reply_text(t(lang, "maintenance"))
        return ConversationHandler.END

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
        [InlineKeyboardButton(t(lang, "btn_singles"), callback_data="game_singles")],
        [InlineKeyboardButton(t(lang, "btn_doubles"), callback_data="game_doubles")],
    ])
    await message.reply_text(t(lang, "ask_game_type"), reply_markup=kb)
    return GAME_TYPE


async def game_singles(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Одиночка — дальше старый поток (цвет/позиция/оба)."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    context.user_data["game_type"] = "singles"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "btn_by_color"), callback_data="id_color")],
        [InlineKeyboardButton(t(lang, "btn_by_position"), callback_data="id_position")],
        [InlineKeyboardButton(t(lang, "btn_dont_know"), callback_data="id_both")],
    ])
    await q.message.reply_text(t(lang, "ask_identify"), reply_markup=kb)
    return IDENTIFY


async def game_doubles(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пара — спрашиваем сторону корта, потом цвет одежды."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    context.user_data["game_type"] = "doubles"
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "btn_near"), callback_data="pos_near")],
        [InlineKeyboardButton(t(lang, "btn_far"), callback_data="pos_far")],
    ])
    await q.message.reply_text(t(lang, "ask_doubles_side"), reply_markup=kb)
    return POSITION


async def identify_color(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь выбрал идентификацию по цвету."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    await q.message.reply_text(t(lang, "ask_shirt"))
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
    await q.message.reply_text(t(lang, "ask_position"), reply_markup=kb)
    return POSITION


async def identify_both(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Пользователь не знает — анализируем обоих/ближнего."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    context.user_data["identify_mode"] = "both"
    await q.message.reply_text(t(lang, "ask_video"))
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
    context.user_data["position"] = "near" if q.data == "pos_near" else "far"

    if context.user_data.get("game_type") == "doubles":
        # Для пары: после стороны уточняем цвет одежды (сузить до игрока)
        context.user_data["identify_mode"] = "doubles"
        await q.message.reply_text(t(lang, "ask_doubles_color"))
        return SHIRT_COLOR
    else:
        # Одиночка по позиции — сразу к видео
        context.user_data["identify_mode"] = "position"
        await q.message.reply_text(t(lang, "ask_video"))
        return VIDEO


def _rotate_frames(frame_paths: list[str], angle: int) -> None:
    """Поворачивает сохранённые кадры на заданный угол (90/180/270)."""
    rot_map = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
               270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    rot_code = rot_map.get(angle)
    if rot_code is None:
        return
    for p in frame_paths:
        try:
            img = cv2.imread(p)
            if img is not None:
                img = cv2.rotate(img, rot_code)
                cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        except Exception:
            continue


async def process_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    lang = db.get_lang(uid)
    name = update.effective_user.first_name

    # Строим описание как найти игрока — зависит от выбранного режима
    mode = context.user_data.get("identify_mode", "both")
    if mode == "doubles":
        # Пара: сторона корта + цвет одежды (сужаем до конкретного игрока)
        pos_key = "pos_near" if context.user_data.get("position") == "near" else "pos_far"
        target_desc = t(lang, "identify_doubles",
                        position=t(lang, pos_key),
                        shirt=context.user_data.get("shirt", "?"))
    elif mode == "color":
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
            url = update.message.text.strip()
            # YouTube блокирует скачивание с серверов — просим файл или Drive
            if "youtube.com" in url or "youtu.be" in url:
                await msg.edit_text(t(lang, "err_youtube"))
                return ConversationHandler.END
            await msg.edit_text(t(lang, "downloading"))
            ok = await loop.run_in_executor(
                None, _download_video_sync, url, video_path)
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
        # АВТООПРЕДЕЛЕНИЕ ОРИЕНТАЦИИ по позе игрока (не зависит от ввода юзера).
        # MediaPipe пробует 4 поворота и выбирает где человек стоит правильно.
        if frames:
            try:
                fpaths = [f["path"] for f in frames]
                best_rot = await loop.run_in_executor(
                    None, pose.detect_best_rotation, fpaths)
                if best_rot:
                    logger.info("Автоповорот кадров: %d°", best_rot)
                    await loop.run_in_executor(
                        None, _rotate_frames, fpaths, best_rot)
            except Exception as e:
                logger.warning("Автоповорот пропущен: %s", e)
        if len(frames) < 3:
            logger.warning("Мало кадров (%d) для user %s", len(frames), uid)
            await msg.edit_text(t(lang, "err_frames"))
            return ConversationHandler.END

        # Проверяем что это РЕАЛЬНО бадминтон — до списания кредита.
        # Защита от мусорных видео и слива денег на галлюцинации.
        await msg.edit_text(t(lang, "validating"))
        is_badminton = await loop.run_in_executor(None, _validate_badminton, frames)
        if not is_badminton:
            logger.info("Не бадминтон — отклонено для user %s", uid)
            await msg.edit_text(t(lang, "err_not_badminton"))
            return ConversationHandler.END

        # Списываем кредит ТОЛЬКО когда уверены что анализ пойдёт.
        # Админ не платит.
        if not _is_admin(uid):
            if not db.consume_credit(uid):
                await msg.edit_text(t(lang, "no_credits"))
                await update.message.reply_text(
                    t(lang, "buy_text"), reply_markup=buy_keyboard(lang))
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

        # Игрок с указанным цветом/позицией не найден — возвращаем кредит,
        # честно сообщаем вместо выдуманного анализа.
        if report == "TARGET_NOT_FOUND":
            if not _is_admin(uid):
                db.add_credits(uid, 1)
            await msg.edit_text(t(lang, "err_target_not_found"))
            await update.message.reply_text(
                t(lang, "menu_hint"), reply_markup=menu_keyboard(lang))
            return ConversationHandler.END

        if report == "NOT_ENOUGH_DATA":
            # Игрок не опознан в кадрах — честный отказ, возврат кредита,
            # вместо пустого отчёта.
            if not _is_admin(uid):
                db.add_credits(uid, 1)
            await msg.edit_text(t(lang, "err_not_enough_data"))
            await update.message.reply_text(
                t(lang, "menu_hint"), reply_markup=menu_keyboard(lang))
            return ConversationHandler.END

        # PDF
        await msg.edit_text(t(lang, "generating"))
        # Генерируем визуалы CV: скелет + тепловую карту + метрики
        skeleton_path = None
        heatmap_path = None
        extra_metrics = None
        try:
            frame_paths = [f["path"] for f in frames]
            sk = f"{tmpdir}/skeleton.jpg"
            skeleton_path = await loop.run_in_executor(
                None, pose.render_skeleton, frame_paths, sk)
            # Метрики (для тепловой карты и блока цифр)
            m = await loop.run_in_executor(None, pose.analyze_frames, frame_paths)
            if m:
                extra_metrics = m
                centers = m.get("_centers")
                if centers:
                    hm = f"{tmpdir}/heatmap.jpg"
                    heatmap_path = await loop.run_in_executor(
                        None, pose.render_heatmap, centers, hm)
        except Exception as e:
            logger.warning("Визуалы CV не отрисованы: %s", e)

        pdf_path = f"{tmpdir}/report.pdf"
        await loop.run_in_executor(
            None, _generate_pdf_sync, report, name, len(frames), lang, pdf_path,
            skeleton_path, heatmap_path, extra_metrics)

        db.log_analysis(uid, len(frames), target_desc[:100])
        remaining = db.get_credits(uid)

        with open(pdf_path, "rb") as f:
            await update.message.reply_document(
                document=f,
                filename=f"RallyIQ_{name}_{datetime.now().strftime('%d%m%Y')}.pdf",
                caption=t(lang, "done", name=name, credits=remaining),
                reply_markup=menu_keyboard(lang),
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
        t(lang, "cancelled"), reply_markup=menu_keyboard(lang))
    return ConversationHandler.END



# ==================================================
# АДМИН-КОМАНДЫ (только для ADMIN_ID)
# ==================================================
async def admin_panel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главная админ-панель."""
    if not _is_admin(update.effective_user.id):
        return
    text = (
        "🛠 Админ-панель RallyIQ\n\n"
        "/stats — статистика проекта\n"
        "/users — последние пользователи\n"
        "/give user_id кол-во — начислить кредиты\n"
        "/take user_id кол-во — забрать кредиты\n"
        "/giveme кол-во — начислить себе\n\n"
        "Ты администратор — все анализы бесплатны."
    )
    await update.message.reply_text(text)


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
    lines = ["👥 Последние пользователи:\n"]
    for u in users:
        uname = f"@{u['username']}" if u["username"] else "—"
        lines.append(
            f"ID: {u['user_id']} | {u['first_name'] or ''} {uname}\n"
            f"   💳 {u['credits']} | 🎬 {u['analyses_count']} | 🌐 {u['lang']}"
        )
    # Без Markdown — имена юзеров могут содержать спецсимволы (_ * `),
    # которые ломают парсинг. Простой текст надёжнее.
    await update.message.reply_text("\n".join(lines))


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


async def admin_take(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/take <user_id> <amount> — забрать кредиты (если начислил по ошибке)."""
    if not _is_admin(update.effective_user.id):
        return
    args = context.args
    if len(args) != 2:
        await update.message.reply_text("Использование: /take <user_id> <кол-во>")
        return
    try:
        target_id = int(args[0])
        amount = int(args[1])
    except ValueError:
        await update.message.reply_text("user_id и кол-во должны быть числами.")
        return
    new_balance = db.take_credits_by_id(target_id, amount)
    if new_balance is None:
        await update.message.reply_text(f"❌ Пользователь {target_id} не найден.")
        return
    await update.message.reply_text(
        f"✅ Списано {amount} анализов у {target_id}.\n"
        f"Новый баланс: {new_balance}"
    )
    try:
        await context.bot.send_message(
            target_id,
            f"⚠️ Баланс изменён. Текущий баланс: {new_balance} анализов."
        )
    except Exception:
        pass


async def admin_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/broadcast <текст> — рассылка сообщения всем пользователям."""
    if not _is_admin(update.effective_user.id):
        return
    text = " ".join(context.args) if context.args else ""
    if not text:
        await update.message.reply_text(
            "Использование: /broadcast <текст>\n\n"
            "Пример: /broadcast 🔧 Через 5 минут короткое обновление (~3 мин). Спасибо за терпение!"
        )
        return

    user_ids = db.get_all_user_ids()
    sent = 0
    failed = 0
    await update.message.reply_text(f"📢 Начинаю рассылку для {len(user_ids)} пользователей...")
    for uid in user_ids:
        try:
            await context.bot.send_message(uid, text)
            sent += 1
        except Exception:
            failed += 1  # юзер заблокировал бота или удалил чат
        # Небольшая пауза чтобы не упереться в лимиты Telegram
        await asyncio.sleep(0.05)
    await update.message.reply_text(
        f"✅ Рассылка завершена.\nДоставлено: {sent}\nНе доставлено: {failed}"
    )


async def admin_maintenance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/maintenance on|off — режим обслуживания (бот отвечает что занят)."""
    if not _is_admin(update.effective_user.id):
        return
    arg = (context.args[0].lower() if context.args else "")
    if arg == "on":
        db.set_setting("maintenance", "1")
        await update.message.reply_text(
            "🔧 Режим обслуживания ВКЛЮЧЕН.\n"
            "Пользователи получат сообщение что бот обновляется.\n"
            "Не забудь выключить: /maintenance off"
        )
    elif arg == "off":
        db.set_setting("maintenance", "0")
        await update.message.reply_text("✅ Режим обслуживания ВЫКЛЮЧЕН. Бот работает обычно.")
    else:
        status = db.get_setting("maintenance", "0")
        state = "ВКЛЮЧЕН 🔧" if status == "1" else "выключен ✅"
        await update.message.reply_text(
            f"Режим обслуживания сейчас: {state}\n\n"
            "Использование: /maintenance on  или  /maintenance off"
        )


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


async def menu_router(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Ловит нажатия кнопок постоянного меню (текст) и направляет в нужный handler.
    Работает на всех языках — сравнивает с локализованными подписями кнопок.
    """
    uid = update.effective_user.id
    lang = db.get_lang(uid)
    text = (update.message.text or "").strip()

    # Сопоставляем нажатую кнопку с действием на любом из языков
    if text in (TEXTS["ru"]["menu_analyze"], TEXTS["kz"]["menu_analyze"], TEXTS["en"]["menu_analyze"]):
        return await analyze_entry_cmd(update, context)
    if text in (TEXTS["ru"]["menu_buy"], TEXTS["kz"]["menu_buy"], TEXTS["en"]["menu_buy"]):
        return await buy(update, context)
    if text in (TEXTS["ru"]["menu_balance"], TEXTS["kz"]["menu_balance"], TEXTS["en"]["menu_balance"]):
        return await balance(update, context)
    if text in (TEXTS["ru"]["menu_language"], TEXTS["kz"]["menu_language"], TEXTS["en"]["menu_language"]):
        return await language_command(update, context)
    if text in (TEXTS["ru"]["menu_faq"], TEXTS["kz"]["menu_faq"], TEXTS["en"]["menu_faq"]):
        return await faq_command(update, context)
    # Не кнопка меню — игнорируем (или подсказываем)
    await update.message.reply_text(t(lang, "menu_hint"), reply_markup=menu_keyboard(lang))


async def faq_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает FAQ с разделами."""
    lang = db.get_lang(update.effective_user.id)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "faq_btn_drive"), callback_data="faq_drive")],
        [InlineKeyboardButton(t(lang, "faq_btn_video"), callback_data="faq_video")],
        [InlineKeyboardButton(t(lang, "faq_btn_how"), callback_data="faq_how")],
        [InlineKeyboardButton(t(lang, "faq_btn_time"), callback_data="faq_time")],
    ])
    await update.message.reply_text(t(lang, "faq_main"), reply_markup=kb)


async def faq_section(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Показывает конкретный раздел FAQ."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    section = q.data.replace("faq_", "")  # drive/video/how/time
    text = t(lang, f"faq_{section}")
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "faq_btn_back"), callback_data="faq_back")],
    ])
    await q.edit_message_text(text, parse_mode="Markdown", reply_markup=kb)


async def faq_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Возврат к списку разделов FAQ."""
    q = update.callback_query
    await q.answer()
    lang = db.get_lang(q.from_user.id)
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton(t(lang, "faq_btn_drive"), callback_data="faq_drive")],
        [InlineKeyboardButton(t(lang, "faq_btn_video"), callback_data="faq_video")],
        [InlineKeyboardButton(t(lang, "faq_btn_how"), callback_data="faq_how")],
        [InlineKeyboardButton(t(lang, "faq_btn_time"), callback_data="faq_time")],
    ])
    await q.edit_message_text(t(lang, "faq_main"), reply_markup=kb)


# ==================================================
# ЗАПУСК
# ==================================================

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Логирует все необработанные ошибки вместо краша."""
    logger.error("Необработанная ошибка: %s", context.error, exc_info=context.error)


async def post_init(app: Application) -> None:
    # Оставляем в синем меню команд только /start — вся навигация через
    # постоянную клавиатуру внизу. Так нет дублирования двух панелей.
    from telegram import BotCommand
    await app.bot.set_my_commands([
        BotCommand("start", "🏸 Start / Restart"),
    ])
    logger.info("Команды меню установлены (минимальные)")


def main() -> None:
    db.init_pool()
    db.init_schema()
    # Проверка MediaPipe: логирует встал ли он. Не ломает бота при сбое.
    pose.log_status()

    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("analyze", analyze_entry_cmd),
            CallbackQueryHandler(analyze_entry_btn, pattern="^go_analyze$"),
            # Кнопка "Анализ" в постоянном меню (на всех языках)
            MessageHandler(
                filters.Regex(f"^({TEXTS['ru']['menu_analyze']}|{TEXTS['kz']['menu_analyze']}|{TEXTS['en']['menu_analyze']})$"),
                analyze_entry_cmd,
            ),
        ],
        states={
            GAME_TYPE: [
                CallbackQueryHandler(game_singles, pattern="^game_singles$"),
                CallbackQueryHandler(game_doubles, pattern="^game_doubles$"),
            ],
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
    app.add_handler(CommandHandler("language", language_command))
    app.add_handler(CommandHandler("buy", buy))
    app.add_handler(CommandHandler("free", free_analysis))
    app.add_handler(CommandHandler("balance", balance))
    # Админские команды
    app.add_handler(CommandHandler("admin", admin_panel))
    app.add_handler(CommandHandler("stats", admin_stats))
    app.add_handler(CommandHandler("users", admin_users))
    app.add_handler(CommandHandler("give", admin_give))
    app.add_handler(CommandHandler("take", admin_take))
    app.add_handler(CommandHandler("giveme", admin_giveme))
    app.add_handler(CommandHandler("broadcast", admin_broadcast))
    app.add_handler(CommandHandler("maintenance", admin_maintenance))
    app.add_handler(conv)
    # Роутер кнопок меню — ловит buy/balance/language/help вне диалога
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, menu_router))
    app.add_handler(CallbackQueryHandler(lang_callback, pattern="^lang_"))
    app.add_handler(CallbackQueryHandler(see_demo_callback, pattern="^see_demo$"))
    app.add_handler(CallbackQueryHandler(buy_callback, pattern="^go_buy$"))
    # Оплата через Telegram Stars
    app.add_handler(CallbackQueryHandler(kaspi_callback, pattern="^pay_kaspi$"))
    app.add_handler(CallbackQueryHandler(send_invoice_callback, pattern="^buy_pack_"))
    app.add_handler(PreCheckoutQueryHandler(precheckout_callback))
    app.add_handler(MessageHandler(filters.SUCCESSFUL_PAYMENT, successful_payment_callback))
    app.add_handler(CallbackQueryHandler(noop_callback, pattern="^noop$"))
    app.add_handler(CallbackQueryHandler(faq_back, pattern="^faq_back$"))
    app.add_handler(CallbackQueryHandler(faq_section, pattern="^faq_(drive|video|how|time)$"))

    app.add_error_handler(error_handler)
    logger.info("RallyIQ запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
