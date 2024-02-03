import torch
import timm
import torchmetrics
import argparse
import os
import time
import json
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

from utils import get_etl_test_data, calculate_precision_recall
from dataset import Dataset

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", type=int, default=0)
args = parser.parse_args()

device = torch.device("cuda:%d"%(args.gpu) if torch.cuda.is_available() else "cpu")

use_class = "../information/hira_kana_kanji.txt"
with open(use_class) as f:
    use_list = [s.strip() for s in f.readlines()]

use_class = "../information/hira_kana_kanji_val.txt"
with open(use_class) as f:
    val_class = [s.strip() for s in f.readlines()]

test_list=get_etl_test_data(use_list)
print("test", len(test_list))


model = timm.create_model("resnet152", pretrained=False, num_classes=len(use_list))

config = {"batch_size":100
        ,"img_size":224
        ,"aug":False
        ,"mean":model.default_cfg["mean"]
        ,"std":model.default_cfg["std"]}


test_data = Dataset(use_list, test_list, config, "test")
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = int(config["batch_size"]), shuffle=False, num_workers=2)


mood_base_dir = "result"
mood_dict = {"etl":"",
             "etl_aug":"",
             "etl_gen_adain_font_to_hand":"",
             "etl_gen_adain_font_to_hand_aug":"",
             "etl_gen_adain_with_clip_font_to_hand":"",
             "etl_gen_adain_with_clip_font_to_hand_aug":"",
             "etl_gen_adain_with_embedding_font_to_hand_aug":"",
             "etl_gen_adain_with_embedding_font_to_hand_aug":"",
             "etl_gen_cdiff_font_to_hand_aug":"",
             "etl_gen_cdiff_font_to_hand_aug":""}



metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(use_list))

for mood in mood_dict:
    print(mood)
    model_path = mood_dict[mood]
    weight = torch.load(model_path, map_location=device)
    model.load_state_dict(weight)
    model = model.to(device)
    model.eval()

    text_path = os.path.join(mood_base_dir, mood, "result_test.txt")
    report_path = os.path.join(mood_base_dir, mood, "report_test.txt")

    pred_list = []
    label_list = []
    for batch in tqdm(test_dataloader):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        preds = model(images)
        preds = torch.argmax(preds.softmax(dim=-1), dim=-1)
        preds = list(preds.to("cpu").detach().numpy())
        labels = list(labels.to("cpu").detach().numpy())

        pred_list+=preds
        label_list+=labels

    # report = classification_report(label_list, pred_list, target_names=use_list, digits=4)
    w = "　        | precision | recall | f1-score | support"
    report = [w]
    f1_list = []
    for idx, c in enumerate(val_class):
        label_idx = use_list.index(c)

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
    with open(report_path, mode="w") as f:
        f.write("\n".join(report))

    write_list = ["{} {} {}".format(test_list[i], use_list[label_list[i]], use_list[pred_list[i]]) for i in range(len(test_list))]
    with open(text_path, mode="w") as f:
        f.write("\n".join(write_list))