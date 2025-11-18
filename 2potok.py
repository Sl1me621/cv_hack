import cv2
import numpy as np

def detect_stop_sign_simple(frame):
    """
    Упрощенная версия обнаружения знака
    """
    # Пороги красного цвета
    lower_red1 = np.array([0, 127, 203])
    upper_red1 = np.array([12, 255, 238])
    lower_red2 = np.array([170, 127, 203])
    upper_red2 = np.array([179, 255, 238])
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Маска красного
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Фильтрация
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    # Поиск контуров
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 500 < area < 500000:
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.7:
                    return True
    return False

import cv2
import numpy as np

def detect_line_direction(opt_frame, debug=False):
    """
    Определяет направление движения относительно черной линии на оптической камере.
    Возвращает:
        direction: "left", "right" или "none"
        offset_norm: нормализованное смещение [-1..1] (отрицательное – линия слева, положительное – справа)
    """

    # --- 1. Предобработка ---
    # Работаем только с нижней частью кадра (там обычно находится линия)
    h, w, _ = opt_frame.shape
    roi_y_start = int(h * 0.6)  # берем нижние 40% кадра
    roi = opt_frame[roi_y_start:, :]

    # В градации серого + размытие для уменьшения шума
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 2. Бинаризация ---
    # Используем Otsu, чтобы автоматически подбирать порог под освещение
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # --- 3. Морфология (чистим маску) ---
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # --- 4. Поиск контуров ---
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "none", 0.0

    # Фильтрация по площади, чтобы убрать мелкий мусор
    roi_area = binary.shape[0] * binary.shape[1]
    min_area = roi_area * 0.01  # игнорируем контуры меньше 1% от площади ROI
    candidates = [c for c in contours if cv2.contourArea(c) > min_area]

    if not candidates:
        return "none", 0.0

    # --- 5. Выбор "похожего на линию" контура ---
    def line_score(cnt):
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch + 1e-6
        aspect = max(cw, ch) / (min(cw, ch) + 1e-6)  # вытянутость
        fill = cv2.contourArea(cnt) / area          # насколько контур заполняет прямоугольник
        # чем вытянутее и плотнее контур – тем выше score
        return aspect * fill

    largest_contour = max(candidates, key=line_score)

    # --- 6. Точный центр по моментам ---
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return "none", 0.0

    cx = int(M["m10"] / M["m00"])  # x-координата центра в ROI
    line_center_x = cx  # уже в координатах ROI (по x совпадает с кадром)

    # Центр кадра по x
    frame_center_x = w // 2

    # --- 7. Нормализованное смещение и направление ---
    offset_px = line_center_x - frame_center_x
    offset_norm = offset_px / frame_center_x  # в диапазоне примерно [-1..1]

    # Порог чувствительности – 5% ширины кадра
    threshold = 0.05

    if offset_norm < -threshold:
        direction = "left"
    elif offset_norm > threshold:
        direction = "right"
    else:
        direction = "none"

    # --- 8. Отладочный вывод (по желанию) ---
    if debug:
        debug_frame = opt_frame.copy()
        # рисуем линию и центр
        cv2.line(debug_frame, (frame_center_x, 0), (frame_center_x, h), (255, 0, 0), 2)
        cv2.circle(debug_frame, (line_center_x, roi_y_start + binary.shape[0] // 2), 5, (0, 0, 255), -1)
        cv2.putText(
            debug_frame,
            f"{direction}, offset={offset_norm:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        # можно показать окно, если запускаешь не на дроне:
        # cv2.imshow("debug", debug_frame)
        # cv2.imshow("binary", binary)
        # cv2.waitKey(1)

    return direction


def process_dual_videos(main_video_path, opt_video_path, output_file="generated_check.txt"):
    """
    Обрабатывает два видео параллельно и возвращает статистику
    """
    main_cap = cv2.VideoCapture(main_video_path)
    opt_cap = cv2.VideoCapture(opt_video_path)
    
    # Получаем общее количество кадров в видео
    total_frames = int(main_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    stop_detected_frame = 0
    stop_found = False
    
    # Список для хранения разметки
    markup_data = []
    
    # Статистика для навигации
    navigation_stats = {
        "left": 0,
        "right": 0, 
        "none": 0,
        "total_processed": 0
    }
    
    print(f"Всего кадров в видео: {total_frames}")
    print("Запуск параллельной обработки двух камер...")
    print("Нажмите 'q' для выхода")
    
    while True:
        ret_main, main_frame = main_cap.read()
        ret_opt, opt_frame = opt_cap.read()
        
        if not ret_main or not ret_opt:
            break
        
        frame_count += 1
        
        # Обработка основной камеры - обнаружение знака
        is_stop = False
        if frame_count / total_frames > 0.9:  # Проверяем знак только после 80% кадров
            is_stop = detect_stop_sign_simple(main_frame)
        
        # Обработка оптической камеры - определение направления
        direction = detect_line_direction(opt_frame)
        navigation_stats[direction] += 1
        navigation_stats["total_processed"] += 1
        
        # Сохраняем данные в разметку
        markup_data.append(f"{frame_count} {direction}")
        
        # Отображаем информацию на объединенном кадре
        main_display = main_frame.copy()
        opt_display = opt_frame.copy()
        
        # Информация на основной камере
        cv2.putText(main_display, f"Main Cam - Frame: {frame_count}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if is_stop:
            stop_found = True
            stop_detected_frame = frame_count
            cv2.putText(main_display, "STOP SIGN DETECTED!", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            status = "Searching..." if frame_count / total_frames > 0.8 else "Waiting 80%..."
            cv2.putText(main_display, f"Stop Sign: {status}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Информация на оптической камере
        cv2.putText(opt_display, f"Optical Cam - Direction: {direction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(opt_display, f"Progress: {frame_count/total_frames*100:.1f}%", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Объединяем кадры для отображения
        if main_display.shape != opt_display.shape:
            # Приводим к одинаковому размеру если нужно
            opt_display = cv2.resize(opt_display, (main_display.shape[1], main_display.shape[0]))
        
        combined = np.vstack([main_display, opt_display])
        cv2.imshow('Dual Camera Processing', combined)
        
        # Выход при обнаружении знака или нажатии 'q'
        if stop_found or cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    main_cap.release()
    opt_cap.release()
    cv2.destroyAllWindows()
    
    # Сохраняем разметку в файл
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
    Основная функция для обработки кадров (аналог функции из eval.py)
    Возвращает направление и флаг обнаружения знака
    """
    # Определяем направление по оптической камере
    direction = detect_line_direction(opt_img)
    
    # Обнаружение знака на основной камере
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