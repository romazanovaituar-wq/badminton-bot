"""
pose.py — изолированный модуль проверки и работы с MediaPipe Pose.

ВАЖНО (этап Variant A): сейчас этот модуль ТОЛЬКО проверяет что MediaPipe
успешно установился и загружается в окружении Railway. Он НЕ подключён к
анализу видео — это безопасная проверка, чтобы убедиться что библиотека
не ломает деплой. Реальный pose-анализ добавим следующим шагом, когда
подтвердим что установка прошла.

Модуль написан так, что любая ошибка импорта/инициализации НЕ роняет бота:
функция is_available() просто вернёт False, и основной пайплайн продолжит
работать как раньше.
"""

import logging

logger = logging.getLogger("rallyiq.pose")

# Пытаемся импортировать mediapipe безопасно
_MP_AVAILABLE = False
_MP_ERROR = None

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except Exception as e:  # noqa: BLE001 — намеренно ловим всё, чтобы не упасть
    _MP_ERROR = str(e)


def is_available() -> bool:
    """Возвращает True если MediaPipe успешно загружен."""
    return _MP_AVAILABLE


def check_pose_init() -> tuple[bool, str]:
    """
    Пробует инициализировать модель позы MediaPipe.
    Возвращает (успех, сообщение). Используется для проверки на старте.
    Ничего не ломает при ошибке.
    """
    if not _MP_AVAILABLE:
        return False, f"MediaPipe не установлен: {_MP_ERROR}"
    try:
        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        )
        pose.close()
        return True, "MediaPipe Pose инициализирован успешно"
    except Exception as e:  # noqa: BLE001
        return False, f"MediaPipe загружен, но Pose не инициализируется: {e}"


def log_status() -> None:
    """Логирует статус MediaPipe при старте бота (для проверки на Railway)."""
    ok, msg = check_pose_init()
    if ok:
        logger.info("✅ POSE CHECK: %s", msg)
    else:
        logger.warning("⚠️ POSE CHECK: %s (бот работает в обычном режиме)", msg)
