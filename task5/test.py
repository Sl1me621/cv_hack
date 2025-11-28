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
    main_video_path = "main/video/main--effiroom.ru.mp4"
    opt_video_path = "main/video/opt--effiroom.ru.mp4"
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