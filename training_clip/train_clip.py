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
import itertools
from sklearn.metrics import accuracy_score

from utils import get_etl_train_val_data, compute_losses
from dataset import Dataset,clip_collate_func
from model import CLIPDualEncoderModel

def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-m", "--model", type=str, default="resnet50")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("-batch", "--batch_size", type=int, default=32)
    parser.add_argument("-opt", "--optimizer", type=str, default="Adam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-size", "--img_size", type=int, default=224)
    parser.add_argument('--aug', action='store_true')

    parser.add_argument("--use_char", type=str, default="../information/hira_kana_kanji.txt")
    parser.add_argument("--element_json", type=str, default="../information/structure.json")
    parser.add_argument("--vocabulary_list", type=str, default="../information/vocabulary.txt")

    parser.add_argument("--image_weight", type=str, default="experiment/train_image_encoder_model-resnet50/model/best_classifier_model.pt")
    parser.add_argument("--text_weight", type=str, default="experiment/train_text_encoder_model-BERT/model/best_classifier_model.pt")

    args = parser.parse_args()
    config = vars(args)
    config["command"] = " ".join(sys.argv)

    device = torch.device("cuda:%d"%(config["gpu"]) if torch.cuda.is_available() else "cpu")

    with open(config["use_char"]) as f:
        use_list = [s.strip() for s in f.readlines()]
    print("num class : ", len(use_list))

    # ファイル全体をリストとして読み込み
    with open(config["vocabulary_list"]) as f:
        vocabulary_list = [s.strip() for s in f.readlines()]
    vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

    model = CLIPDualEncoderModel(device=device, 
                                 char_size=len(use_list),
                                 image_encoder_alias=config["model"], 
                                 vocabulary_size=len(vocabulary_list), 
                                 image_embedding_dims=2048,
                                 image_encoder_weight=config["image_weight"],
                                 text_encoder_weight=config["text_weight"])
    model.to(device)
    config["mean"] = model.image_encoder.model.default_cfg["mean"]
    config["std"] = model.image_encoder.model.default_cfg["std"]

    run_name = "train_clip-{}-BERT".format(config["model"])
    save_log = os.path.join("experiment", run_name, "log")
    save_model = os.path.join("experiment", run_name, "model")
    os.makedirs(save_log, exist_ok=True)
    os.makedirs(save_model, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=save_log)
    with open(os.path.join("experiment", run_name, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    train_list, val_list = get_etl_train_val_data(use_list)
    train_data = Dataset(use_list, train_list, config, "train")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size = config["batch_size"], shuffle=True, num_workers=8, collate_fn=clip_collate_func)

    test_data = Dataset(use_list, val_list, config, "val")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size = int(config["batch_size"]/4), shuffle=False, num_workers=4, collate_fn=clip_collate_func)

    weight_decay = 0.0
    head_lr = 1e-3
    lr_scheduler_patience = 1.0
    lr_scheduler_factor = 0.8

    parameters = [
        {"params": model.image_encoder.parameters(), "lr": config["learning_rate"]},
        {"params": model.text_encoder.parameters(), "lr": config["learning_rate"]},
        {
            "params": itertools.chain(
                model.image_projection.parameters(),
                model.text_projection.parameters(),
            ),
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
    ]

    if config["optimizer"]=="SGD":
        optimizer = torch.optim.SGD(parameters, weight_decay=weight_decay)
    elif config["optimizer"]=="Adam":
        optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay)
    elif config["optimizer"]=="AdamW":
        optimizer = torch.optim.AdamW(parameters, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=lr_scheduler_patience,
        factor=lr_scheduler_factor,
    )

    train_step = 0
    test_step = 0
    best_loss = 1000
    for epoch in range(config["epochs"]):
        model.train()#学習モードに移行
        s_time = time.time()

        for batch in train_dataloader:
            image, tokens_tensor,token_type_ids_tensor,attention_mask_tensor = batch
            image, tokens_tensor,token_type_ids_tensor,attention_mask_tensor = image.to(device), tokens_tensor.to(device),token_type_ids_tensor.to(device),attention_mask_tensor.to(device)
            inputs = {"input_ids":tokens_tensor, "token_type_ids":token_type_ids_tensor, "attention_mask":attention_mask_tensor}

            optimizer.zero_grad()

            image_embeddings, text_embeddings = model(image,inputs)
            loss = compute_losses(image_embeddings, text_embeddings).mean()

            summary_writer.add_scalar("train_loss", float(loss.to('cpu').detach().numpy().copy()), train_step)

            loss.backward()
            optimizer.step()
            train_step+=1

        test_loss = 0
        for batch in test_dataloader:
            image, tokens_tensor,token_type_ids_tensor,attention_mask_tensor = batch
            image, tokens_tensor,token_type_ids_tensor,attention_mask_tensor = image.to(device), tokens_tensor.to(device),token_type_ids_tensor.to(device),attention_mask_tensor.to(device)
            inputs = {"input_ids":tokens_tensor, "token_type_ids":token_type_ids_tensor, "attention_mask":attention_mask_tensor}

            image_embeddings, text_embeddings = model(image,inputs)
            loss = compute_losses(image_embeddings, text_embeddings).mean()

            summary_writer.add_scalar("val_loss", float(loss.to('cpu').detach().numpy().copy()), test_step)
            test_loss+=loss.to('cpu').detach().numpy().copy()
            test_step+=1

        test_loss/=len(test_dataloader)
        lr_scheduler.step(test_loss)

        summary_writer.add_scalar("loss", test_loss, epoch)

        print("Time for epoch {} is {} | Loss {}".format(epoch, time.time() - s_time, test_loss))

        if test_loss <=best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), os.path.join(save_model, "best_clip_model.pt"))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_model, "clip_model_ep%s.pt"%(epoch+1)))

if __name__ == "__main__":
    main()