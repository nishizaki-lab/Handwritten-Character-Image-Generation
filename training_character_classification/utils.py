import os
import random

import json
def get_etl_train_val_data(char_list):
    train_list = []
    val_list = []

    path = "../data_paths/ETL_train/all_paths.json"
    with open(path) as f:
        train_d = json.load(f)

    path = "../data_paths/ETL_val/all_paths.json"
    with open(path) as f:
        val_d = json.load(f)

    for char in char_list:
        if char in train_d:
            train_list+=["{} {}".format(char, s) for s in train_d[char]]

        if char in val_d:
            val_list+=["{} {}".format(char, s) for s in val_d[char]]

    return train_list, val_list

def get_etl_test_data(char_list):
    test_list = []

    path = "../data_paths/ETL_test/all_paths.json"
    with open(path) as f:
        test_d = json.load(f)

    for char in char_list:
        if char in test_d:
            test_list+=["{} {}".format(char, s) for s in test_d[char]]
    return test_list

def get_gen_data(use_gen, char_list, num_use=None):
    gen_list = []
    path = "{}/all_paths.json".format(use_gen)
    with open(path) as f:
        gen_d = json.load(f)

    if "cDiff" in use_gen.split("/")[-1]:
        for char in char_list:
            char_paths = ["{} {}".format(char, s) for s in gen_d[char]]
            if len(char_paths)>=200:
                char_paths=char_paths[:200]
            gen_list+=char_paths
    else:
        for char in char_list:
            char_paths = ["{} {}".format(char, s) for s in gen_d[char]]
            if num_use is not None:
                char_paths = random.sample(char_paths, num_use)
            gen_list+=char_paths
    return gen_list

from sklearn.metrics import classification_report,accuracy_score
import numpy as np

def calculate_precision_recall(array1, array2, target_value):
    TP = np.sum((array1 == target_value) & (array2 == target_value))
    FP = np.sum((array1 != target_value) & (array2 == target_value))
    FN = np.sum((array1 == target_value) & (array2 != target_value))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall

def get_classification_report(all_char, chars, label_list, pred_list):
    w = "　        | precision | recall | f1-score | support"
    report = [w]
    f1_list = []
    for idx, c in enumerate(chars):
        label_idx = all_char.index(c)

        precision, recall = calculate_precision_recall(np.array(label_list), np.array(pred_list), label_idx)
        f1 = (2*precision*recall)/(precision+recall) if (precision+recall) !=0 else 0.0

        w = "       {} |    {:.4f} | {:.4f} |   {:.4f} |     {}".format(c, precision, recall, f1, label_list.count(label_idx))
        report.append(w)
        f1_list.append(f1)

    macro_F1 = sum(f1_list)/len(f1_list)
    accuracy = accuracy_score(label_list, pred_list)
    w = ""
    report.append(w)
    w = "accuracy                           {:.4f} |  {}".format(accuracy, len(label_list))
    report.append(w)
    w = "macro f1                           {:.4f} |  {}".format(macro_F1, len(label_list))
    report.append(w)

    return report




import cv2

def crop_all_contours(image):
    # しきい値処理で物体を分離
    _, thresholded = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # 輪郭を検出
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 全ての輪郭の座標を結合
        # all_contours = np.vstack(contours[i] for i in range(len(contours)))
        all_contours = np.vstack([contours[i] for i in range(len(contours))])

        # 全ての輪郭を囲む長方形を取得
        x, y, w, h = cv2.boundingRect(all_contours)

        # 画像を切り抜く
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    else:
        return image

def adjust_square(img, blank=0):
    h, w = img.shape[0], img.shape[1]
    img = cv2.bitwise_not(img)
    if h>=w:
        img = cv2.copyMakeBorder(img, 0, 0, int((h-w)/2), int((h-w)/2), cv2.BORDER_CONSTANT, 0)
    else:
        img = cv2.copyMakeBorder(img, int((w-h)/2), int((w-h)/2), 0, 0, cv2.BORDER_CONSTANT, 0)
    # 正方形にする
    img = cv2.copyMakeBorder(img, blank, blank, blank, blank, cv2.BORDER_CONSTANT, 0)
    img = cv2.bitwise_not(img)
    img = cv2.resize(img, (img.shape[0], img.shape[0]))

    return img

def known_pad(img, pad=20, pad_value=0):
    h, w = img.shape[0], img.shape[1]
    img = cv2.bitwise_not(img)
    up_pad, down_pad, left_pad, right_pad = pad, pad, pad, pad
    # 正方形にする
    img = cv2.copyMakeBorder(img, up_pad, down_pad, right_pad, left_pad, cv2.BORDER_CONSTANT, pad_value)
    img = cv2.bitwise_not(img)
    return img