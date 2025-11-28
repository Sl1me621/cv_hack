import cv2
import numpy as np
from eval import video_process

def load_task(path):
    task_dict = {}
    with open(path, "r") as f:
        task = f.readlines()
        for line in task:
            number, direction = line.replace("\n", "").split(" ")
            task_dict[int(number)] = direction
    return task_dict

if __name__ == "__main__":
    main_cam = cv2.VideoCapture("main/video/main.avi")
    
    opr_cam = cv2.VideoCapture("main/video/opt.avi")
    
    task_dict = load_task("main\check.txt")

    frame_number = 0
    true_count = 0
    total_count = 0
    stop_found = False
    
    while True:
        ret, main_img = main_cam.read()
        if not ret:
            print("no ret")
            break
        ret2, opt_img = opr_cam.read()
        if not ret2:
            print("no ret2")
            break

        direction, is_stop = video_process(main_img, opt_img)
        # print(direction,is_stop)
        
        if 152 <= frame_number <= 1000:
            if 770 <= frame_number <= 1000 and is_stop:
                stop_found = True
                break
            
            if frame_number in task_dict and direction == task_dict[frame_number]:
                true_count += 1
            total_count += 1
        
        frame_number += 1

    accuracy = (true_count / total_count * 100) if total_count > 0 else 0
    
    if accuracy > 80 and stop_found:
        print(accuracy)
        print("Решение верное")
    else:
        print(accuracy)
        print(true_count)
        print("Решение неверное")
    
