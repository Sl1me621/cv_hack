from pioneer_sdk2 import Pioneer, Camera, CameraType, Event
import cv2
import numpy as np
import threading


# ============ ЛОГИКА ОБРАБОТКИ ЛИНИИ ============
def detect_direction(opt_frame):
    if opt_frame is None:
        return "none"

    h, w = opt_frame.shape[:2]
    cx = w // 2

    roi = opt_frame[h // 2:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    col_sum = bw.sum(axis=0)
    if col_sum.max() < 10:
        return "none"

    x = int(np.argmax(col_sum))

    if abs(x - cx) < 20:
        return "none"

    return "left" if x < cx else "right"


# ========== ОБНАРУЖЕНИЕ ЗНАКА ВЪЕЗД ЗАПРЕЩЁН ==========
def detect_stop(main_frame):
    if main_frame is None:
        return False

    img = main_frame
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 80, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in conts:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if 0.7 < w / h < 1.3:
            roi = img[y:y+h, x:x+w]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 40, 255])
            white_mask = cv2.inRange(roi_hsv, lower_white, upper_white)

            if np.count_nonzero(white_mask) > (roi.size * 0.02):
                return True

    return False


# ========== ОЖИДАНИЕ ВЗЛЁТА ==========
takeoff_complete = False

def on_takeoff(_):
    global takeoff_complete
    takeoff_complete = True


# ========== ОСНОВНОЙ ЦИКЛ ==========
def main():
    global takeoff_complete

    main_cam = Camera(CameraType.MAIN)
    opt_cam = Camera(CameraType.OPT)
    drone = Pioneer()

    drone.subscribe(on_takeoff, Event.TAKEOFF_COMPLETE)

    # Ждём взлёта
    while not takeoff_complete:
        pass

    # После взлёта — только вывод направления и флага
    while True:
        main_frame = main_cam.get_cv_frame()
        opt_frame = opt_cam.get_cv_frame()

        if main_frame is None or opt_frame is None:
            break

        direction = detect_direction(opt_frame)
        stop_flag = detect_stop(main_frame)

        print(direction, stop_flag)

        if stop_flag:
            break


if __name__ == "__main__":
    main()
