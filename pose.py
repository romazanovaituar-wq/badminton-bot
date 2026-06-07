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
    leg_stretches = [] # растяжка ног (расстояние лодыжка-лодыжка = глубина выпада)
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
                    # Растяжка ног: полное расстояние между лодыжками (выпад)
                    leg_stretch = ((la[0]-ra[0])**2 + (la[1]-ra[1])**2) ** 0.5
                    leg_stretches.append(leg_stretch)

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
        "leg_stretch": round(avg(leg_stretches), 3) if leg_stretches else None,
        "total_distance": round(sum(
            ((centers[i][0]-centers[i-1][0])**2 + (centers[i][1]-centers[i-1][1])**2) ** 0.5
            for i in range(1, len(centers))
        ) * 100, 1) if len(centers) >= 2 else 0.0,
        "_centers": centers,  # координаты для тепловой карты (служебное)
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
    ls = m.get("leg_stretch")
    if ls is not None:
        if ls > 0.25:
            parts.append("широкие выпады (хорошая растяжка ног)")
        elif ls < 0.1:
            parts.append("узкая стойка, мало выпадов")
    return "; ".join(parts)


def compute_scores(m: dict) -> dict | None:
    """
    Вычисляет баллы (0-100) НАПРЯМУЮ из измеренных метрик.
    Воспроизводимо: одно видео = один балл.

    Калибровка основана на биомеханических исследованиях бадминтона:
    - профи эффективно сгибают колени (не обязательно глубже, а функциональнее)
    - активность/скорость перемещений — маркер уровня
    - вынос руки и собранный корпус = техника
    Потолок поднят до 98, чтобы профи получали честные 88-95.
    """
    if not m:
        return None

    def clamp(v, lo=25, hi=98):
        return max(lo, min(hi, int(round(v))))

    scores = {}

    # FOOTWORK: функциональный сгиб колена (110-160° рабочий диапазон) + активность.
    # Щедрее: рабочее колено даёт базу ~80, активность добавляет до +18.
    knee = m.get("knee_flex")
    mv = m.get("movement", 0) or 0
    if knee is not None:
        if 110 <= knee <= 165:
            knee_score = 80 - abs(140 - knee) * 0.4
        elif knee > 165:
            knee_score = 68 - (knee - 165) * 1.3        # прямые ноги = штраф
        else:
            knee_score = 72 - (110 - knee) * 0.5
        move_bonus = min(16, mv * 2.0)
        scores["footwork"] = clamp(knee_score + move_bonus)

    # TECHNIQUE: угол локтя (замах) + вынос руки вверх. Щедрая база.
    elbow = m.get("elbow_angle")
    arm = m.get("arm_up_ratio")
    if elbow is not None:
        elbow_score = 76 - abs(110 - elbow) * 0.45
        if arm is not None:
            elbow_score += arm * 20   # рука поднята для ударов сверху = техничнее
        scores["technique"] = clamp(elbow_score)

    # POSITIONING: собранный корпус. Небольшой наклон вперёд нормален для готовности.
    lean = m.get("torso_lean")
    if lean is not None:
        pos_score = 84 - max(0, lean - 18) * 1.6
        scores["positioning"] = clamp(pos_score)

    # RECOVERY/баланс: симметрия плеч + активность (быстрое восстановление).
    bal = m.get("balance")
    if bal is not None:
        rec_score = 86 - bal * 2.8
        rec_score += min(8, mv * 1.0)
        scores["recovery"] = clamp(rec_score)

    return scores if scores else None


def level_from_scores(scores: dict) -> str:
    """
    Определяет уровень игрока по измеренным баллам (англ. ключи).
    Используется чтобы GPT не занижал всех до 'любителя'.
    """
    if not scores:
        return ""
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    if not vals:
        return ""
    avg = sum(vals) / len(vals)
    if avg >= 88:
        return "профессиональный"
    if avg >= 78:
        return "продвинутый"
    if avg >= 65:
        return "уверенный любитель"
    if avg >= 50:
        return "любитель"
    return "начинающий"


def render_skeleton(frame_paths: list[str], out_path: str) -> str | None:
    """
    Берёт кадр где поза найдена наиболее уверенно, рисует на нём скелет
    MediaPipe (точки суставов + соединения) и сохраняет картинку.
    Возвращает путь к картинке или None если не удалось.

    Это визуальное доказательство работы CV — вставляется в PDF.
    """
    if not _MP_AVAILABLE:
        return None
    try:
        import cv2
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        mp_styles = mp.solutions.drawing_styles
    except Exception as e:
        logger.warning("render_skeleton: импорт не удался: %s", e)
        return None

    best_img = None
    best_landmarks = None
    best_score = -1.0

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
                    # Оцениваем "качество" позы — средняя видимость точек
                    vis = [lm.visibility for lm in res.pose_landmarks.landmark]
                    score = sum(vis) / len(vis) if vis else 0
                    if score > best_score:
                        best_score = score
                        best_img = img.copy()
                        best_landmarks = res.pose_landmarks
                except Exception:
                    continue

        if best_img is None or best_landmarks is None:
            return None

        # Рисуем скелет на лучшем кадре
        mp_draw.draw_landmarks(
            best_img,
            best_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(
                color=(90, 190, 240), thickness=3, circle_radius=4),
            connection_drawing_spec=mp_draw.DrawingSpec(
                color=(0, 200, 150), thickness=3),
        )
        cv2.imwrite(out_path, best_img)
        return out_path
    except Exception as e:
        logger.warning("render_skeleton: ошибка: %s", e)
        return None


def render_heatmap(centers: list, out_path: str) -> str | None:
    """
    Рисует тепловую карту перемещений игрока по корту на основе его
    позиций (центров) в кадрах. Показывает где игрок проводил больше времени.
    Возвращает путь к картинке или None.

    centers — список (x, y) нормализованных координат (0..1) от MediaPipe.
    """
    if not centers or len(centers) < 3:
        return None
    try:
        import numpy as np
        import cv2
    except Exception as e:
        logger.warning("render_heatmap: импорт не удался: %s", e)
        return None

    try:
        W, H = 360, 480  # вертикальный корт
        # Пустое поле (тёмный фон под тему PDF)
        canvas = np.full((H, W, 3), (33, 27, 21), dtype=np.uint8)  # BGR ~ #121621

        # Рисуем разметку корта (простую)
        line_color = (90, 90, 90)
        cv2.rectangle(canvas, (30, 30), (W-30, H-30), line_color, 2)
        cv2.line(canvas, (30, H//2), (W-30, H//2), (70, 120, 160), 2)  # сетка
        cv2.line(canvas, (W//2, 30), (W//2, H-30), line_color, 1)       # центр

        # Накапливаем тепло в сетке
        heat = np.zeros((H, W), dtype=np.float32)
        for (x, y) in centers:
            px = int(30 + x * (W - 60))
            py = int(30 + y * (H - 60))
            px = max(0, min(W-1, px))
            py = max(0, min(H-1, py))
            cv2.circle(heat, (px, py), 28, 1.0, -1)

        # Размытие для плавности
        heat = cv2.GaussianBlur(heat, (0, 0), 18)
        if heat.max() > 0:
            heat = heat / heat.max()

        # Накладываем цветовую карту (синий→жёлтый→красный)
        heat_u8 = (heat * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        # Смешиваем с полем там где есть тепло
        mask = (heat > 0.05).astype(np.float32)[..., None]
        canvas = (canvas * (1 - mask * 0.65) + colored * (mask * 0.65)).astype(np.uint8)

        # Перерисуем линии поверх
        cv2.rectangle(canvas, (30, 30), (W-30, H-30), line_color, 2)
        cv2.line(canvas, (30, H//2), (W-30, H//2), (70, 120, 160), 2)

        cv2.imwrite(out_path, canvas)
        return out_path
    except Exception as e:
        logger.warning("render_heatmap: ошибка: %s", e)
        return None


def detect_upside_down(frame_paths: list[str]) -> bool:
    """
    Определяет перевёрнут ли игрок вверх ногами на кадрах.
    Использует MediaPipe: если нос (голова) НИЖЕ бёдер в кадре —
    значит изображение перевёрнуто на 180°.
    Возвращает True если нужно повернуть на 180°.
    """
    if not _MP_AVAILABLE:
        return False
    try:
        import cv2
        mp_pose = mp.solutions.pose
    except Exception:
        return False

    upside_votes = 0
    normal_votes = 0
    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
        ) as pose:
            for path in frame_paths[:10]:  # хватит 10 кадров для голосования
                try:
                    img = cv2.imread(path)
                    if img is None:
                        continue
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = pose.process(rgb)
                    if not res.pose_landmarks:
                        continue
                    lm = res.pose_landmarks.landmark
                    L = mp_pose.PoseLandmark
                    # Y растёт вниз. Нос должен быть ВЫШЕ (меньше Y) чем бёдра.
                    nose_y = lm[L.NOSE].y
                    hip_y = (lm[L.LEFT_HIP].y + lm[L.RIGHT_HIP].y) / 2
                    # видимость носа и бёдер должна быть приличной
                    if (lm[L.NOSE].visibility > 0.3 and
                            lm[L.LEFT_HIP].visibility > 0.3):
                        if nose_y > hip_y:
                            upside_votes += 1   # нос ниже бёдер = перевёрнут
                        else:
                            normal_votes += 1
                except Exception:
                    continue
    except Exception:
        return False

    # Перевёрнут если большинство кадров за это
    return upside_votes > normal_votes and upside_votes >= 2


def detect_best_rotation(frame_paths: list[str]) -> int:
    """
    Определяет ЛУЧШИЙ поворот кадров по позе игрока.
    Пробует 4 варианта (0, 90, 180, 270) и выбирает где поза человека
    выглядит наиболее естественно: голова сверху, ноги снизу, тело вертикально.
    Возвращает угол поворота (0/90/180/270) который надо применить.

    Это полностью автоматически — не зависит от ввода пользователя.
    """
    if not _MP_AVAILABLE:
        return 0
    try:
        import cv2
        mp_pose = mp.solutions.pose
    except Exception:
        return 0

    # Берём несколько кадров для надёжности
    sample = frame_paths[:8]
    rotations = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
                 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    scores = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}

    try:
        with mp_pose.Pose(
            static_image_mode=True, model_complexity=1,
            min_detection_confidence=0.5,
        ) as pose:
            for path in sample:
                img0 = cv2.imread(path)
                if img0 is None:
                    continue
                for angle, rot_code in rotations.items():
                    img = img0 if rot_code is None else cv2.rotate(img0, rot_code)
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res = pose.process(rgb)
                    if not res.pose_landmarks:
                        continue
                    lm = res.pose_landmarks.landmark
                    L = mp_pose.PoseLandmark
                    nose = lm[L.NOSE]
                    lhip, rhip = lm[L.LEFT_HIP], lm[L.RIGHT_HIP]
                    lank, rank = lm[L.LEFT_ANKLE], lm[L.RIGHT_ANKLE]
                    # Качество "правильной" ориентации:
                    # нос выше бёдер, бёдра выше лодыжек (человек стоит)
                    vis = (nose.visibility + lhip.visibility + rhip.visibility) / 3
                    if vis < 0.3:
                        continue
                    hip_y = (lhip.y + rhip.y) / 2
                    ank_y = (lank.y + rank.y) / 2
                    s = 0.0
                    if nose.y < hip_y:      # голова выше бёдер
                        s += 1.0
                    if hip_y < ank_y:        # бёдра выше лодыжек
                        s += 1.0
                    s *= vis
                    scores[angle] += s
    except Exception:
        return 0

    best = max(scores, key=scores.get)
    # Если лучший вариант 0 или все нули — не поворачиваем
    if scores[best] == 0:
        return 0
    return best
