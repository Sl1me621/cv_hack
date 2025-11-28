# -*- coding: utf-8 -*-
import cv2
import pandas as pd
import eval as submission

"""Файл служит для определения точности вашего алгоритма.
   Не редактируёте его!!!
   Для получения оценки точности, запустите файл на исполнение.
"""

DETECTION_THRESHOLD = 0.75


def IoU(user_box, true_box):
    """IoU = Area of overlap / Area of union
       Output: 0.0 ... 1.0
    """

    x1 = max(user_box[0], true_box[0])
    y1 = max(user_box[1], true_box[1])
    x2 = min(user_box[2], true_box[2])
    y2 = min(user_box[3], true_box[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (user_box[2] - user_box[0] + 1) * (user_box[3] - user_box[1] + 1)
    box2_area = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def check_answer(expected, detected):
    if len(detected) != len(expected):
        return False
    if IoU(expected, detected) < DETECTION_THRESHOLD:
        return False
    return True


def main():
    user_data_list = submission.load_models()

    csv_file = "annotations_test.csv"
    data = pd.read_csv(csv_file, sep=';', encoding='utf-8')
    data = data.sample(frac=1)

    correct = 0
    for row in data.itertuples():
        _, image_filename, true_res = row
        true_res = true_res[1:]
        image = cv2.imread(image_filename)
        user_res = submission.detect_cars(image, user_data_list)
        true_res = [int(i) for i in true_res.split()]

        if check_answer(user_res, true_res):
            correct += 1

    total_object = len(data.index)
    print(f"Для {correct} изображений из {total_object} автомобили детектированы верно.")
    score = round(correct / total_object, 2)
    print(f"Точность: {score}")

if __name__ == '__main__':
    main()
