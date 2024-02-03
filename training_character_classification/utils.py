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
