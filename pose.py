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


# ==================================================
# РЕАЛЬНЫЙ POSE-АНАЛИЗ (измерения вместо догадок)
# ==================================================
import math


def _angle(a, b, c) -> float:
    """Угол в точке b между отрезками b-a и b-c, в градусах (0-180)."""
    try:
        ang = math.degrees(
            math.atan2(c[1] - b[1], c[0] - b[0])
            - math.atan2(a[1] - b[1], a[0] - b[0])
        )
        ang = abs(ang)
        if ang > 180:
            ang = 360 - ang
        return ang
    except Exception:
        return 0.0


def analyze_frames(frame_paths: list[str]) -> dict | None:
    """
    Прогоняет кадры через MediaPipe Pose и считает реальные метрики движения.
    Возвращает словарь с измерениями или None если поза нигде не найдена.

    Метрики:
      - knee_flex: средний угол сгиба колена (меньше = глубже присед = лучше готовность)
      - elbow_angle: средний угол локтя бьющей руки
      - stance_width: средняя ширина стойки (расстояние между лодыжками)
      - torso_lean: средний наклон корпуса от вертикали
      - movement: средняя величина перемещения между кадрами (активность)
      - frames_with_pose: в скольких кадрах найден человек
    """
    if not _MP_AVAILABLE:
        return None

    try:
        import cv2
        mp_pose = mp.solutions.pose
    except Exception as e:
        logger.warning("pose.analyze_frames: импорт не удался: %s", e)
        return None

    knee_angles = []
    elbow_angles = []
    stance_widths = []
    torso_leans = []
    centers = []  # центр массы (бёдра) для оценки движения
    arm_reaches = []   # вынос бьющей руки вверх (запястье выше плеча)
    balances = []      # симметрия плеч (горизонтальность = баланс)
    found = 0

    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        ) as pose:
            for path in frame_paths:
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = pose.process(rgb)
                    if not res.pose_landmarks:
                        continue
                    found += 1
                    lm = res.pose_landmarks.landmark
                    L = mp_pose.PoseLandmark

                    def pt(landmark):
                        return (lm[landmark].x, lm[landmark].y)

                    # Угол колена (правая нога): бедро-колено-лодыжка
                    knee = _angle(pt(L.RIGHT_HIP), pt(L.RIGHT_KNEE), pt(L.RIGHT_ANKLE))
                    if knee > 0:
                        knee_angles.append(knee)

                    # Угол локтя (правая рука): плечо-локоть-запястье
                    elbow = _angle(pt(L.RIGHT_SHOULDER), pt(L.RIGHT_ELBOW), pt(L.RIGHT_WRIST))
                    if elbow > 0:
                        elbow_angles.append(elbow)

                    # Ширина стойки: расстояние между лодыжками
                    la = pt(L.LEFT_ANKLE)
                    ra = pt(L.RIGHT_ANKLE)
                    stance = abs(la[0] - ra[0])
                    stance_widths.append(stance)

                    # Наклон корпуса: угол линии плечи-бёдра от вертикали
                    sh_mid = ((pt(L.LEFT_SHOULDER)[0] + pt(L.RIGHT_SHOULDER)[0]) / 2,
                              (pt(L.LEFT_SHOULDER)[1] + pt(L.RIGHT_SHOULDER)[1]) / 2)
                    hip_mid = ((pt(L.LEFT_HIP)[0] + pt(L.RIGHT_HIP)[0]) / 2,
                               (pt(L.LEFT_HIP)[1] + pt(L.RIGHT_HIP)[1]) / 2)
                    # отклонение по горизонтали относительно вертикали
                    dx = abs(sh_mid[0] - hip_mid[0])
                    dy = abs(sh_mid[1] - hip_mid[1]) + 1e-6
                    lean = math.degrees(math.atan2(dx, dy))
                    torso_leans.append(lean)

                    # Вынос руки вверх: насколько запястье выше плеча
                    # (в координатах экрана меньше Y = выше). Положительное = рука поднята.
                    wrist_y = pt(L.RIGHT_WRIST)[1]
                    shoulder_y = pt(L.RIGHT_SHOULDER)[1]
                    reach = (shoulder_y - wrist_y)  # >0 если рука выше плеча
                    arm_reaches.append(reach)

                    # Баланс: насколько плечи горизонтальны (разница высот плеч).
                    # Меньше разница = ровнее стойка = лучше баланс.
                    sh_diff = abs(pt(L.LEFT_SHOULDER)[1] - pt(L.RIGHT_SHOULDER)[1])
                    balances.append(sh_diff)

                    centers.append(hip_mid)
                except Exception:
                    continue
    except Exception as e:
        logger.warning("pose.analyze_frames: ошибка обработки: %s", e)
        return None

    if found < 3:
        # Слишком мало кадров с позой — данные ненадёжны
        return None

    # Среднее перемещение центра между последовательными кадрами
    movement = 0.0
    if len(centers) >= 2:
        dists = []
        for i in range(1, len(centers)):
            d = math.hypot(centers[i][0] - centers[i-1][0],
                           centers[i][1] - centers[i-1][1])
            dists.append(d)
        movement = sum(dists) / len(dists) if dists else 0.0

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else None

    # Вынос руки: доля кадров где рука поднята выше плеча (хорошо для ударов сверху)
    arm_up_ratio = None
    if arm_reaches:
        arm_up_ratio = round(sum(1 for r in arm_reaches if r > 0) / len(arm_reaches), 2)

    # Баланс: средняя разница высот плеч (меньше = ровнее). В процентах.
    balance = round(avg(balances) * 100, 1) if balances else None

    return {
        "frames_with_pose": found,
        "knee_flex": avg(knee_angles),
        "elbow_angle": avg(elbow_angles),
        "stance_width": round(avg(stance_widths) * 100, 1) if stance_widths else None,
        "torso_lean": avg(torso_leans),
        "movement": round(movement * 100, 2),
        "arm_up_ratio": arm_up_ratio,
        "balance": balance,
    }


def metrics_summary(m: dict, lang: str = "ru") -> str:
    """
    Превращает метрики в краткий текст для подсказки GPT.
    Это РЕАЛЬНЫЕ измерения — GPT будет на них опираться вместо догадок.
    """
    if not m:
        return ""
    parts = []
    knee = m.get("knee_flex")
    if knee is not None:
        # Прямое колено (>165) = плохая готовность; согнутое (<150) = хорошая
        if knee > 165:
            parts.append(f"колени почти прямые (угол {knee}°, слабая готовность к рывку)")
        elif knee < 150:
            parts.append(f"колени хорошо согнуты (угол {knee}°, хорошая стойка)")
        else:
            parts.append(f"умеренный сгиб колен (угол {knee}°)")
    lean = m.get("torso_lean")
    if lean is not None:
        if lean > 25:
            parts.append(f"сильный наклон корпуса ({lean}°)")
        else:
            parts.append(f"корпус относительно вертикален ({lean}°)")
    mv = m.get("movement")
    if mv is not None:
        if mv < 2:
            parts.append("мало перемещений между кадрами (низкая активность ног)")
        elif mv > 6:
            parts.append("активные перемещения по корту")
        else:
            parts.append("умеренная активность перемещений")
    return "; ".join(parts)


def compute_scores(m: dict) -> dict | None:
    """
    Вычисляет баллы (0-100) НАПРЯМУЮ из измеренных метрик.
    Это делает оценки ВОСПРОИЗВОДИМЫМИ — одно видео всегда даёт один балл,
    потому что баллы выводятся из физических измерений, а не из догадок.

    Возвращает баллы по тем категориям, что можно измерить позой.
    Тактику не считаем — её скелетом не измеришь (оставляем GPT).
    """
    if not m:
        return None

    def clamp(v):
        return max(20, min(95, int(round(v))))

    scores = {}

    # FOOTWORK (работа ног): согнутые колени + активность перемещений.
    # Идеальный сгиб колена в бадминтоне ~120-150° (готовность к рывку).
    knee = m.get("knee_flex")
    mv = m.get("movement", 0) or 0
    if knee is not None:
        # 120-150° = отлично (90), прямые >170 = плохо (40)
        if knee <= 150:
            knee_score = 90 - abs(135 - knee) * 0.8
        else:
            knee_score = 90 - (knee - 150) * 1.8  # штраф за прямые ноги
        # активность добавляет до +15
        move_bonus = min(15, mv * 2)
        scores["footwork"] = clamp(knee_score + move_bonus - 7)

    # TECHNIQUE (техника): вынос руки + угол локтя + наклон корпуса.
    elbow = m.get("elbow_angle")
    arm = m.get("arm_up_ratio")
    if elbow is not None:
        # согнутый локоть при замахе (~90-140) хорошо
        elbow_score = 85 - abs(115 - elbow) * 0.6
        if arm is not None:
            elbow_score += arm * 15  # рука часто поднята = +
        scores["technique"] = clamp(elbow_score)

    # POSITIONING (позиционирование): наклон корпуса (вертикальный = собран).
    lean = m.get("torso_lean")
    if lean is not None:
        # умеренный наклon 5-20° норм, сильный >30 плохо
        pos_score = 88 - max(0, lean - 18) * 1.5
        scores["positioning"] = clamp(pos_score)

    # RECOVERY (восстановление/баланс): симметрия + стабильность стойки.
    bal = m.get("balance")
    if bal is not None:
        # меньше разница плеч = ровнее = лучше. bal в процентах.
        rec_score = 90 - bal * 2.5
        scores["recovery"] = clamp(rec_score)

    return scores if scores else None
