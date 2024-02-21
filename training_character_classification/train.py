import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import timm
import argparse
import sys
import os
import json
import time
from sklearn.metrics import accuracy_score


from utils import get_etl_train_val_data, get_gen_data, get_classification_report
from dataset import Dataset

BASE_NUM_ETL_DATA = 611768

def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    # 使用モデル
    # vit_base_patch8_224
    # resnet152(現状固定)
    parser.add_argument("-m", "--model", type=str, default="resnet152")
    # 事前学習済みモデルを使うか
    parser.add_argument("--pretrained", action="store_true")
    # 使用する生成画像のパスの一覧ディレクトリ
    parser.add_argument("--use_gen", type=str, default="")
    # 生成画像選定を行った場合には，何枚使うか
    parser.add_argument("--num_use", type=str, default=None)
    # データ拡張を使うか
    parser.add_argument("--aug", action="store_true")
    # 活性化関数(現状固定)
    parser.add_argument("-opt", "--optimizer", type=str, default="Adam")
    # 学習率(現状固定)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    # バッチサイズ(現状固定)
    parser.add_argument("-batch", "--batch_size", type=int, default=64)
    # 画像サイズ(現状固定)
    parser.add_argument("-size", "--img_size", type=int, default=224)
    args = parser.parse_args()
    config = vars(args)
    config["command"] = " ".join(sys.argv)

    device = torch.device("cuda:%d"%(config["gpu"]) if torch.cuda.is_available() else "cpu")

    use_class = "../information/hira_kana_kanji.txt"
    with open(use_class) as f:
        use_list = [s.strip() for s in f.readlines()]

    use_class = "../information/hira_kana_kanji_val.txt"
    with open(use_class) as f:
        val_class = [s.strip() for s in f.readlines()]

    print("num class : ", len(use_list))

    train_list, val_list = get_etl_train_val_data(use_list)
    if len(config["use_gen"]) != 0:
        use_gen_name= "_{}".format(config["use_gen"].split("/")[-1])
        if config["num_use"] is not None:
            num_use = "_use-{}".format(int(config["num_use"]))
            gen_list=get_gen_data(config["use_gen"], use_list, num_use=int(config["num_use"]))
        else:
            num_use = "_use-all"
            gen_list=get_gen_data(config["use_gen"], use_list)
    else:
        use_gen_name= ""
        num_use = ""
        gen_list = []

    print("train etl", len(train_list))
    print("train gen", len(gen_list))
    print("train", len(train_list)+len(gen_list))
    print("val", len(val_list))

    train_list = train_list+gen_list

    epochs = int(100*((BASE_NUM_ETL_DATA/len(train_list))))
    print("epochs", epochs)
    if epochs < 1:
        epochs = 10
    config["epochs"] = epochs

    run_name = "etl{}_model-{}_pre-{}_aug-{}{}".format(use_gen_name,
                                                        config["model"],
                                                        config["pretrained"],
                                                        config["aug"],
                                                        num_use)

    save_log = os.path.join("experiment", run_name, "log")
    save_model = os.path.join("experiment", run_name, "model")
    os.makedirs(save_log, exist_ok=True)
    os.makedirs(save_model, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=save_log)
    with open(os.path.join("experiment", run_name, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    model = timm.create_model(config["model"], pretrained=config["pretrained"], num_classes=len(use_list))
    model = model.to(device)
    config["mean"] = model.default_cfg["mean"]
    config["std"] = model.default_cfg["std"]

    train_data = Dataset(use_list, train_list, config, "train")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = config["batch_size"], shuffle=True, num_workers=4)

    test_data = Dataset(use_list, val_list, config, "val")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = int(config["batch_size"]/4), shuffle=False, num_workers=1)

    if config["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"]=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(use_list))
    criterion = nn.CrossEntropyLoss()

    train_step = 0
    test_step = 0
    global_step = 0
    best_acc = 0

    for epoch in range(epochs):
        model.train()#学習モードに移行
        s_time = time.time()

        for batch in train_dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(images)

            loss = criterion(preds, labels)

            summary_writer.add_scalar("train_loss", float(loss.to("cpu").detach().numpy().copy()), train_step)

            preds = preds.softmax(dim=-1)
            preds = preds.to("cpu").detach()
            labels = labels.to("cpu").detach()
            acc = metric(preds, labels)
            summary_writer.add_scalar("train_acc", acc, train_step)

            loss.backward()
            optimizer.step()

            if train_step%int(BASE_NUM_ETL_DATA/config["batch_size"])==0:
                model.eval()#学習モードに移行
                test_loss = 0
                test_preds = []
                test_labels = []
                for batch in test_dataloader:
                    images, labels = batch
                    images, labels = images.to(device), labels.to(device)

                    preds = model(images)

                    loss = criterion(preds, labels)

                    summary_writer.add_scalar("val_loss", float(loss.to("cpu").detach().numpy().copy()), test_step)
                    test_loss+=float(loss.to("cpu").detach().numpy().copy())

                    preds = torch.argmax(preds.softmax(dim=-1), dim=-1)
                    preds = list(preds.to("cpu").detach().numpy())
                    labels = list(labels.to("cpu").detach().numpy())
                    acc = accuracy_score(preds, labels)
                    summary_writer.add_scalar("val_acc", acc, test_step)
                    test_step+=1
                    test_preds+=preds
                    test_labels+=labels

                test_loss/=len(test_dataloader)
                accuracy = accuracy_score(test_labels, test_preds)

                summary_writer.add_scalar("acc", accuracy, global_step)
                summary_writer.add_scalar("loss", test_loss, global_step)

                print("Time for iter {} is {} | Val Acc {}".format(global_step, time.time() - s_time, accuracy))
                if accuracy >=best_acc:
                    best_acc = accuracy
                    torch.save(model.state_dict(), os.path.join(save_model, "best_classifier_model.pt"))

                    report = get_classification_report(use_list, val_class, test_labels, test_preds)
                    with open(os.path.join("experiment", run_name, "report.txt"), mode="w") as f:
                        f.write("\n".join(report))

                if global_step % 1 == 0:
                    torch.save(model.state_dict(), os.path.join(save_model, "classifier_model_iter%s.pt"%(global_step)))

                global_step+=1
                s_time = time.time()
                model.train()

            train_step+=1

    torch.save(model.state_dict(), os.path.join(save_model, "classifier_model.pt"))

if __name__ == "__main__":
    main()