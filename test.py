import cv2
import numpy as np

def detect_stop_sign_simple(frame, debug=False):
    """
    Упрощенное обнаружение знака STOP:
    - По красному цвету в HSV
    - По размеру контура
    - По форме (многоугольник, близкий к восьмиугольнику)
    """

    # Пороги красного цвета (можно потом подкрутить под свои видео)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([12, 127, 203])
    upper_red2 = np.array([179, 255, 238])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Маска красного
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask2

    # Морфология для очистки
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    frame_area = h * w

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.001 or area > frame_area * 0.3:
            continue  # слишком маленькие или слишком огромные

        # Аппроксимация многоугольником
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertices = len(approx)

        # STOP – обычно 8-угольник, но даём допуск
        if 6 <= vertices <= 10:
            x, y, w_box, h_box = cv2.boundingRect(contour)
            aspect_ratio = w_box / float(h_box)
            if 0.7 < aspect_ratio < 1.3:  # почти квадратный
                if debug:
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    cv2.putText(frame, "STOP?", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("stop_debug", frame)
                    cv2.waitKey(1)
                return True

    return False

import cv2
import numpy as np
from collections import deque

def detect_line_direction(opt_frame, debug=False):
    """
    Определяет направление движения относительно черной линии.
    Возвращает:
        direction: "left", "right" или "none"
        offset_norm: [-1..1], отрицательное – линия слева, положительное – справа
    """
    h, w, _ = opt_frame.shape

    # --- 1. ROI: нижняя часть кадра ---
    roi_y_start = int(h * 0.6)  # нижние 40% кадра
    roi = opt_frame[roi_y_start:, :]
    roi_h, roi_w = roi.shape[:2]

    # --- 2. Grayscale + CLAHE + blur ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 3. Бинаризация: чёрная линия -> белое ---
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- 4. Морфология ---
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # --- 5. Компоненты связности ---
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    if num_labels <= 1:
        # только фон
        return "none", 0.0

    # stats[0] — фон, остальные — объекты
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = np.argmax(areas)
    largest_label = max_idx + 1
    largest_area = areas[max_idx]

    roi_area = roi_h * roi_w
    min_area = roi_area * 0.025  # игнорируем слишком маленькие области (<1.5% ROI)
    if largest_area < min_area:
        return "none", 0.0

    # Геометрия компоненты
    x_obj = stats[largest_label, cv2.CC_STAT_LEFT]
    y_obj = stats[largest_label, cv2.CC_STAT_TOP]
    w_obj = stats[largest_label, cv2.CC_STAT_WIDTH]
    h_obj = stats[largest_label, cv2.CC_STAT_HEIGHT]

    # Требуем, чтобы линия была достаточно высокой (вытянутая)
    if h_obj < roi_h * 0.15:  # меньше 20% высоты ROI — считаем шумом
        return "none", 0.0

    # Маска только крупнейшего компонента
    line_mask = (labels == largest_label).astype(np.uint8)

    # --- 6. Проверка: линия должна доходить до нижней части ROI ---
    bottom_band_start = int(roi_h * 0.8)  # нижние 20% ROI
    bottom_band = line_mask[bottom_band_start:, :]
    bottom_pixels = np.count_nonzero(bottom_band)

    # Требуем, чтобы внизу было достаточно пикселей линии
    min_bottom_pixels = int(roi_w * 0.02)  # хотя бы 2% ширины в пикселях
    if bottom_pixels < min_bottom_pixels:
        # линия где-то выше, внизу её нет -> считаем, что для движения не релевантно
        return "none", 0.0

    # --- 7. Центр линии по нижней полосе ---
    ys_bottom, xs_bottom = np.where(bottom_band > 0)
    if len(xs_bottom) == 0:
        return "none", 0.0

    cx_bottom = int(np.mean(xs_bottom))  # X-среднее по нижней части линии

    # переводим в координаты всего кадра
    line_center_x = cx_bottom
    frame_center_x = w // 2

    offset_px = line_center_x - frame_center_x
    offset_norm = offset_px / frame_center_x  # примерно [-1..1]

    # --- 8. Направление с более широкой мёртвой зоной ---
    threshold = 0.05 # 8% ширины кадра, чтобы чаще давать 'none' около центра

    if offset_norm < -threshold:
        direction = "left"
    elif offset_norm > threshold:
        direction = "right"
    else:
        direction = "none"

    # --- 9. Отладка ---
    if debug:
        debug_frame = opt_frame.copy()

        # Центр кадра
        cv2.line(debug_frame, (frame_center_x, 0), (frame_center_x, h), (255, 0, 0), 2)

        # Центр линии внизу ROI
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



def process_dual_videos(main_video_path, opt_video_path, output_file="generated_check.txt",
                        smooth_window=5):
    main_cap = cv2.VideoCapture(main_video_path)
    opt_cap = cv2.VideoCapture(opt_video_path)
    
    total_frames = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    stop_detected_frame = 0
    stop_found = False

    markup_data = []

    navigation_stats = {
        "left": 0,
        "right": 0, 
        "none": 0,
        "total_processed": 0
    }

    # очередь для сглаживания смещения
    offset_history = deque(maxlen=smooth_window)

    # === Новое: счётчик подряд идущих кадров со знаком STOP ===
    stop_streak = 0
    stop_threshold_frames = 5  # сколько подряд кадров нужно, чтобы считать знак действительно найденным

    print(f"Всего кадров в видео: {total_frames}")
    print("Запуск параллельной обработки двух камер...")
    print("Нажмите 'q' для выхода")
    
    while True:
        ret_main, main_frame = main_cap.read()
        ret_opt, opt_frame = opt_cap.read()
        
        if not ret_main or not ret_opt:
            break
        
        frame_count += 1
        
        # ---- 1. Стоп-знак на основной камере ----
        is_stop_raw = False
        if frame_count / total_frames > 0.6:  # после 80% начинаем поиск
            is_stop_raw = detect_stop_sign_simple(main_frame)

        # накапливаем подряд идущие кадры с детектом
        if is_stop_raw:
            stop_streak += 1
        else:
            stop_streak = 0

        # считаем знак найденным, если подряд >= N кадров видим STOP
        is_stop = stop_streak >= stop_threshold_frames

        # один раз фиксируем кадр первого уверенного обнаружения
        if is_stop and not stop_found:
            stop_found = True
            stop_detected_frame = frame_count
        
        # ---- 2. Линия на оптической камере ----
        direction_raw, offset_norm_raw = detect_line_direction(opt_frame)
        offset_history.append(offset_norm_raw)

        # сглаженное смещение
        if len(offset_history) > 0:
            offset_norm_smooth = float(sum(offset_history) / len(offset_history))
        else:
            offset_norm_smooth = offset_norm_raw

        # пересчёт направления по сглаженному
        threshold = 0.055
        if offset_norm_smooth < -threshold:
            direction = "left"
        elif offset_norm_smooth > threshold:
            direction = "right"
        else:
            direction = "none"

        navigation_stats[direction] += 1
        navigation_stats["total_processed"] += 1
        
        # ---- 3. Сохраняем разметку ----
        # формат: кадр, направление, нормализованное смещение (сглаженное)
        markup_data.append(f"{frame_count} {direction} {offset_norm_smooth:.4f}")
        
        # ---- 4. Отрисовка ----
        main_display = main_frame.copy()
        opt_display = opt_frame.copy()
        
        cv2.putText(main_display, f"Main Cam - Frame: {frame_count}/{total_frames}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if is_stop:
            cv2.putText(main_display, "STOP SIGN DETECTED!", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            status = "Searching..." if frame_count / total_frames > 0.8 else "Waiting 80%..."
            cv2.putText(main_display, f"Stop Sign: {status}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        cv2.putText(opt_display, f"Optical Cam - Dir: {direction}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(opt_display, f"Offset: {offset_norm_smooth:.2f}", (10, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(opt_display, f"Progress: {frame_count/total_frames*100:.1f}%", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        if main_display.shape != opt_display.shape:
            opt_display = cv2.resize(opt_display, (main_display.shape[1], main_display.shape[0]))
        
        combined = np.vstack([main_display, opt_display])
        cv2.imshow('Dual Camera Processing', combined)
        
        # выходим либо по уверенно найденному STOP, либо по 'q'
        if stop_found or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    main_cap.release()
    opt_cap.release()
    cv2.destroyAllWindows()
    
    save_markup_to_file(markup_data, output_file, stop_found, stop_detected_frame)
    
    return stop_found, stop_detected_frame, frame_count, total_frames, navigation_stats, markup_data


def save_markup_to_file(markup_data, filename, stop_found, stop_frame):
    """
    Сохраняет разметку в файл в том же формате, что и check.txt
    """
    with open(filename, 'w') as f:
        for line in markup_data:
            f.write(line + '\n')
    
    print(f"\n✅ Разметка сохранена в файл: {filename}")
    print(f"📝 Сохранено записей: {len(markup_data)}")
    if stop_found:
        print(f"🛑 Знак стоп обнаружен на кадре: {stop_frame}")

def video_process(main_img, opt_img):
    """
    Аналог функции из eval.py
    Возвращает:
        direction: "left"/"right"/"none"
        is_stop: True/False
    """
    direction, _ = detect_line_direction(opt_img)
    is_stop = detect_stop_sign_simple(main_img)
    return direction, is_stop


# Основной код
if __name__ == "__main__":
    main_video_path = "video/main--effiroom.ru.mp4"
    opt_video_path = "video/opt--effiroom.ru.mp4"
    output_markup_file = "generated_check.txt"
    
    # Обрабатываем оба видео
    stop_found, stop_frame, processed_frames, total_frames, nav_stats, markup_data = process_dual_videos(
        main_video_path, opt_video_path, output_markup_file
    )
    
    if stop_found:
        frames_remaining = total_frames - stop_frame
        progress_percent = (stop_frame / total_frames) * 100

        print(f" Кадр обнаружения: {stop_frame}")
        print(f" Пройдено кадров: {stop_frame}/{total_frames} ({progress_percent:.1f}%)")
    else:
        print(f" Знак не обнаружен")
        print(f" Обработано кадров: {processed_frames}/{total_frames}")