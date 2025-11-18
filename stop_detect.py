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

# Основной код с статистикой
cap = cv2.VideoCapture("video/main--effiroom.ru.mp4")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
stop_detected_frame = 0
stop_found = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Отображаем информацию на кадре
    display_frame = frame.copy()
    cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, "Press 'q' to quit", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # print(frame_count/total_frames )           
    
    if detect_stop_sign_simple(frame) and frame_count/total_frames >0.8 :
        stop_found = True
        stop_detected_frame = frame_count
        frames_remaining = total_frames - frame_count
        
        # cv2.putText(display_frame, "STOP SIGN DETECTED!", (10, 90), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        # cv2.putText(display_frame, f"Exiting program...", (10, 120), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        break
    
    cv2.imshow('Stop Sign Detection', display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if stop_found:
    frames_remaining = total_frames - stop_detected_frame
    progress_percent = (stop_detected_frame / total_frames) * 100
    
    print(f" Кадр обнаружения: {stop_detected_frame}")
    print(f" Пройдено кадров: {stop_detected_frame}/{total_frames} ({progress_percent:.1f}%)")
else:
    print(f" Знак не обнаружен")
    print(f" Обработано кадров: {frame_count}/{total_frames}")

print("="*50)