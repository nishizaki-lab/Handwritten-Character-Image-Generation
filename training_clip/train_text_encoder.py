import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from transformers import BertForSequenceClassification,BertConfig
import timm
import argparse
import sys
import os
import json
import time
from sklearn.metrics import accuracy_score

from dataset import TextDataset, text_collate_func
from model import BertEmbeddings

def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("-batch", "--batch_size", type=int, default=512)
    parser.add_argument("-opt", "--optimizer", type=str, default="Adam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)

    parser.add_argument("--element_json", type=str, default="../information/structure.json")
    parser.add_argument("--vocabulary_list", type=str, default="../information/vocabulary.txt")

    args = parser.parse_args()
    config = vars(args)
    config["command"] = " ".join(sys.argv)

    device = torch.device("cuda:%d"%(config["gpu"]) if torch.cuda.is_available() else "cpu")

    use_class = "../information/hira_kana_kanji.txt"
    with open(use_class) as f:
        use_list = [s.strip() for s in f.readlines()]
    print("num class : ", len(use_list))

    # ファイル全体をリストとして読み込み
    with open(config["vocabulary_list"]) as f:
        vocabulary_list = [s.strip() for s in f.readlines()]
    vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

    config["model"] = "BERT"
    run_name = "train_text_encoder_model-{}".format(config["model"])
    save_log = os.path.join("experiment", run_name, "log")
    save_model = os.path.join("experiment", run_name, "model")
    os.makedirs(save_log, exist_ok=True)
    os.makedirs(save_model, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=save_log)
    with open(os.path.join("experiment", run_name, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    train_data = TextDataset(use_list, config, "train")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = config["batch_size"], shuffle=True, num_workers=8, collate_fn=text_collate_func)

    test_data = TextDataset(use_list, config, "val")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = int(config["batch_size"]/4), shuffle=False, num_workers=4, collate_fn=text_collate_func)

    configuration = BertConfig(vocab_size = len(vocabulary_list),
                            max_position_embeddings=13,
                            hidden_size=768)
    model_ckpt = "yosuke/bert-base-japanese-char"
    model = BertForSequenceClassification(configuration).from_pretrained(model_ckpt, num_labels=len(use_list))
    bertembeddings = BertEmbeddings(configuration)
    model.bert.embeddings = bertembeddings
    model.to(device)

    if config["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"]=="Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_step = 0
    test_step = 0
    best_acc = 0
    for epoch in range(config["epochs"]):
        model.train()#学習モードに移行
        s_time = time.time()

        for batch in train_dataloader:
            tokens_tensor,token_type_ids_tensor,attention_mask_tensor, labels = batch
            tokens_tensor,token_type_ids_tensor,attention_mask_tensor, labels = tokens_tensor.to(device),token_type_ids_tensor.to(device),attention_mask_tensor.to(device), labels.to(device)
            inputs = {"input_ids":tokens_tensor, "token_type_ids":token_type_ids_tensor, "attention_mask":attention_mask_tensor}

            optimizer.zero_grad()
            loss = model(**inputs,labels=labels).loss

            summary_writer.add_scalar("train_loss", float(loss.to('cpu').detach().numpy().copy()), train_step)

            loss.backward()
            optimizer.step()
            train_step+=1

        model.eval()
        test_loss = 0
        test_preds = []
        test_labels = []
        for batch in test_dataloader:
            tokens_tensor,token_type_ids_tensor,attention_mask_tensor, labels = batch
            tokens_tensor,token_type_ids_tensor,attention_mask_tensor, labels = tokens_tensor.to(device),token_type_ids_tensor.to(device),attention_mask_tensor.to(device), labels.to(device)
            inputs = {"input_ids":tokens_tensor, "token_type_ids":token_type_ids_tensor, "attention_mask":attention_mask_tensor}

            with torch.no_grad():
                preds = model(**inputs).logits
                loss = model(**inputs,labels=labels).loss
                # loss = loss_fct(logits, labels)
            summary_writer.add_scalar("val_loss", float(loss.to("cpu").detach().numpy().copy()), test_step)
            test_loss+=float(loss.to('cpu').detach().numpy().copy())

            preds = torch.argmax(preds.softmax(dim=-1), dim=-1)
            preds = list(preds.to("cpu").detach().numpy())
            labels = list(labels.to("cpu").detach().numpy())
            # print("preds", preds[:10])
            # print("labels", labels[:10])
            acc = accuracy_score(preds, labels)
            summary_writer.add_scalar("val_acc", acc, test_step)
            test_step+=1
            test_preds+=preds
            test_labels+=labels

        test_loss/=len(test_dataloader)
        scheduler.step(test_loss)
        accuracy = accuracy_score(test_labels, test_preds)

        summary_writer.add_scalar("acc", accuracy, epoch)
        summary_writer.add_scalar("loss", test_loss, epoch)

        print("Time for epoch {} is {} | Val Acc {}".format(epoch, time.time() - s_time, accuracy))

        if accuracy >=best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), os.path.join(save_model, "best_classifier_model.pt"))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_model, "classifier_model_ep%s.pt"%(epoch+1)))


if __name__ == "__main__":
    main()