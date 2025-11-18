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
    Определяет направление движения относительно черной линии.
    Возвращает:
        direction: "left", "right" или "none"
        offset_norm: [-1..1], отрицательное – линия слева, положительное – справа
    """
    h, w, _ = opt_frame.shape

    # 1. ROI: нижняя часть кадра
    roi_y_start = int(h * 0.6)
    roi = opt_frame[roi_y_start:, :]
    roi_h, roi_w = roi.shape[:2]

    # 2. Grayscale + CLAHE + blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Бинаризация (чёрная линия -> белое)
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. Морфология
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5. Компоненты связности
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        return "none", 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas)
    largest_label = max_idx + 1
    largest_area = areas[max_idx]

    roi_area = roi_h * roi_w
    min_area = roi_area * 0.025
    if largest_area < min_area:
        return "none", 0.0

    x_obj = stats[largest_label, cv2.CC_STAT_LEFT]
    y_obj = stats[largest_label, cv2.CC_STAT_TOP]
    w_obj = stats[largest_label, cv2.CC_STAT_WIDTH]
    h_obj = stats[largest_label, cv2.CC_STAT_HEIGHT]

    if h_obj < roi_h * 0.15:
        return "none", 0.0

    # Маска линии
    line_mask = (labels == largest_label).astype(np.uint8)

    # Проверка: линия должна доходить до низа
    bottom_band_start = int(roi_h * 0.8)
    bottom_band = line_mask[bottom_band_start:, :]
    bottom_pixels = np.count_nonzero(bottom_band)
    min_bottom_pixels = int(roi_w * 0.02)
    if bottom_pixels < min_bottom_pixels:
        return "none", 0.0

    # Центр по нижней полосе
    ys_bottom, xs_bottom = np.where(bottom_band > 0)
    if len(xs_bottom) == 0:
        return "none", 0.0

    cx_bottom = int(np.mean(xs_bottom))

    line_center_x = cx_bottom
    frame_center_x = w // 2

    offset_px = line_center_x - frame_center_x
    offset_norm = offset_px / frame_center_x

    # Порог
    threshold = 0.055

    if offset_norm < -threshold:
        direction = "left"
    elif offset_norm > threshold:
        direction = "right"
    else:
        direction = "none"

    if debug:
        debug_frame = opt_frame.copy()
        cv2.line(debug_frame, (frame_center_x, 0), (frame_center_x, h), (255, 0, 0), 2)
        cy_full = roi_y_start + bottom_band_start + (bottom_band.shape[0] // 2)
        cv2.circle(debug_frame, (line_center_x, cy_full), 6, (0, 0, 255), -1)
        text = f"{direction}, off={offset_norm:.2f}"
        cv2.putText(
            debug_frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        vis_mask = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.imshow("line_binary", vis_mask)
        cv2.imshow("line_debug", debug_frame)
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
    threshold = 0.055
    if offset_norm_smooth < -threshold:
        direction = "left"
    elif offset_norm_smooth > threshold:
        direction = "right"
    else:
        direction = "none"

    # Детекция знака STOP
    is_stop = detect_stop_sign_simple(main_img)

    return direction, bool(is_stop)
