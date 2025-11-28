import cv2
import os

# Быстрая версия
cap = cv2.VideoCapture("video/opt.avi")  # Укажите путь к видео
output_dir = "captured_frames"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    cv2.imshow('Video', frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):
        filename = f"{output_dir}/frame_{frame_count:04d}.png"
        cv2.imwrite(filename, frame)
        print(f"Сохранено: {filename}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()