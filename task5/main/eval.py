import cv2
import numpy as np
from collections import deque

# ==========================
#   STOP SIGN DETECTION
# ==========================

def detect_stop_sign_simple(frame, debug=False):
    """
    Упрощенное обнаружение знака STOP:
    - По красному цвету в HSV
    - По размеру контура
    - По форме (многоугольник, близкий к восьмиугольнику)
    """
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([12, 127, 203])
    upper_red2 = np.array([179, 255, 238])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Маска красного (используем второй диапазон, т.к. он у тебя уже подкручен под видео)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask2

    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    frame_area = h * w

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.001 or area > frame_area * 0.3:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertices = len(approx)

        if 6 <= vertices <= 10:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            aspect_ratio = w_box / float(h_box)
            if 0.7 < aspect_ratio < 1.3:
                if debug:
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    cv2.putText(frame, "STOP?", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("stop_debug", frame)
                    cv2.waitKey(1)
                return True

    return False


# ==========================
#   LINE DIRECTION DETECTION
# ==========================

def detect_line_direction(opt_frame, debug=False):
    """
    Полностью новый алгоритм:
    - Вертикальная проекция
    - 1D сглаживание
    - Надёжный поиск максимума
    """

    h, w, _ = opt_frame.shape
    center_x = w // 2

    # 1. ROI нижняя часть кадра
    roi_y_start = int(h * 0.45)
    roi = opt_frame[roi_y_start:]
    roi_h, roi_w = roi.shape[:2]

    # 2. Градации серого
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 3. Бинаризация
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. Удаление шумов
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 5. Вертикальная проекция (sum по y)
    col_sum = binary.sum(axis=0).astype(np.float32)

    # Если линии нет
    if col_sum.max() < 50:  # мягкий порог
        return "none", 0.0

    # 6. 1D сглаживание вертикального профиля
    col_sum = cv2.GaussianBlur(col_sum.reshape(-1,1), (9,1), 0).flatten()

    # 7. Поиск глобального максимума
    line_x = int(np.argmax(col_sum))

    # 8. Нормированный offset
    offset_px = line_x - center_x
    offset_norm = offset_px / center_x

    # 9. Порог для направления
    threshold = 0.05

    if offset_norm < -threshold:
        direction = "left"
    elif offset_norm > threshold:
        direction = "right"
    else:
        direction = "none"

    # ===== Debug visualization =====
    if debug:
        dbg = opt_frame.copy()
        cv2.line(dbg, (center_x, 0), (center_x, h), (0, 255, 0), 2)
        cv2.circle(dbg, (line_x, h - 10), 8, (0, 0, 255), -1)
        cv2.putText(dbg, f"{direction} off={offset_norm:.2f}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,255), 2)
        cv2.imshow("line_debug", dbg)
        cv2.imshow("binary", binary)
        cv2.imshow("projection", col_sum.astype(np.uint8))
        cv2.waitKey(1)

    return direction, float(offset_norm)



# ==========================
#   VIDEO PROCESS (API ДЛЯ tester.py)
# ==========================

# Можно добавить лёгкое сглаживание по кадрам
_offset_history = deque(maxlen=5)

def video_process(main_img: np.ndarray, opt_img: np.ndarray):
    """
    Функция, которую вызывает tester.py
    Возвращает:
        direction: "left"/"right"/"none"
        is_stop: True/False
    """
    # Направление
    direction_raw, offset_norm_raw = detect_line_direction(opt_img)

    # Сгладим offset по истории
    _offset_history.append(offset_norm_raw)
    if len(_offset_history) > 0:
        offset_norm_smooth = float(sum(_offset_history) / len(_offset_history))
    else:
        offset_norm_smooth = offset_norm_raw

    # Пересчитаем направление по сглаженному offset
    threshold = 0.05
    if offset_norm_smooth < -threshold:
        direction = "left"
    elif offset_norm_smooth > threshold:
        direction = "right"
    else:
        direction = "none"

    # Детекция знака STOP
    is_stop = detect_stop_sign_simple(main_img)

    return direction, bool(is_stop)
