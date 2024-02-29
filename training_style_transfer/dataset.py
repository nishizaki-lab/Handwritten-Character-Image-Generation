import torchvision.transforms as transforms
import torch
import os
import cv2
import albumentations as A
import random
import glob2
import numpy as np
from PIL import Image, ImageDraw, ImageFont,ImageOps

from utils import *

class Dataset(torch.utils.data.Dataset):

    def __init__(self, config):
        super().__init__()

        self.step = config["step_type"]

        # ファイル全体をリストとして読み込み
        with open(config["use_char"]) as f:
            self.char_list = [s.strip() for s in f.readlines()]

        self.img_size = config["img_size"]

        self.transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]
                            )
                        ])

        self.char_path_dict,_ = get_etl_train_val_data()

        # フォント
        with open(config["use_font"]) as f:
            self.fontfile_paths = [s.strip() for s in f.readlines()]
        self.fontfile = "../information/BIZUDPGothic-Regular.ttf"

        self.indices = self.fontfile_paths*10
        self.len = len(self.indices)

        with open(config["vocabulary_list"]) as f:
            vocabulary_list = [s.strip() for s in f.readlines()]
        self.vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

        self.kanji2element = load_json(config["element_json"])

    def get_tokens(self, char):
        tokens = []
        tokens_tensor = []

        elements = self.kanji2element[char]
        element = random.choice(elements)
        tokens = element.split(" ")
        # print(char, tokens)

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_type_ids = [0]*len(tokens)
        attention_mask = [0 if t == "[PAD]" else 1 for t in tokens]
        tokens_tensor = [self.vocabulary_list.index(t) if t in self.vocabulary_list else 1 for t in tokens]

        token_type_ids_tensor = torch.tensor(token_type_ids)
        attention_mask_tensor = torch.tensor(attention_mask)
        tokens_tensor = torch.tensor(tokens_tensor)

        # inputs = {"input_ids":tokens_tensor, "token_type_ids":token_type_ids_tensor, "attention_mask":attention_mask_tensor}

        return tokens, tokens_tensor,token_type_ids_tensor,attention_mask_tensor

    def get_font_image(self, path, is_multi_font=False, is_random_pad=False, is_rote=False, fontfile=None, rote=None):
        char = path.split("/")[-3]
        if is_multi_font:
            count=0
            while True:
                if fontfile is not None and count==0:
                    pass
                else:
                    fontfile = random.choice(self.fontfile_paths)
                img = char2font(char, fontfile, is_pad=is_random_pad)
                if img is not None:
                    if not is_random_pad:
                        img = known_pad(img)
                    break

                if count==20:
                    img = path2handimg(path, is_aug=is_rote)
                    break
                count+=1
        else:
            img = char2font(char, self.fontfile, is_pad=is_random_pad)
            if not is_random_pad:
                img = known_pad(img)

        # if int(random.uniform(0,1)+0.25) and is_rote:
        if is_rote:
            if rote is None:
                rote = random.randint(-15,15)
            img = rote_img(img, rote)

        return img

    def get_handwrite_image(self, path, is_random_pad=False, is_rote=False, rote=None):
        img = path2handimg(path, is_aug=False)

        if is_random_pad:
            img = random_pad(img)

        if is_rote:
            if rote is None:
                rote = random.randint(-15,15)
            img = rote_img(img, rote)

        return img

    # Content：一つのフォント，位置固定，回転あり
    # Style：複数フォント,位置固定，回転あり
    def step1(self, path, fontfile=None):
        char = path.split("/")[-3]
        label = self.char_list.index(char)
        label = torch.tensor(label)

        if int(random.uniform(0,1)+0.5):
            is_rote=False
            rote = None
        else:
            is_rote=True
            rote = random.randint(-15,15)

        img = self.get_font_image(path, is_multi_font=False, is_random_pad=False, is_rote=is_rote, fontfile=fontfile, rote=rote)
        img = cv2.resize(img,(self.img_size, self.img_size))
        img = self.transforms(img)

        style_img = self.get_font_image(path, is_multi_font=True, is_random_pad=False, is_rote=is_rote, fontfile=fontfile, rote=rote)
        style_img = cv2.resize(style_img,(self.img_size, self.img_size))
        style_img = self.transforms(style_img)

        return img, style_img, label

    # Content：複数フォント，位置自由，回転あり
    # Style：複数フォント, 位置固定，回転あり
    def step2(self, path, fontfile=None):
        char = path.split("/")[-3]
        label = self.char_list.index(char)
        label = torch.tensor(label)

        if int(random.uniform(0,1)+0.5):
            is_rote=False
            rote = None
        else:
            is_rote=True
            rote = random.randint(-15,15)

        img = self.get_font_image(path, is_multi_font=True, is_random_pad=False, is_rote=is_rote, fontfile=fontfile, rote=rote)
        img = cv2.resize(img,(self.img_size, self.img_size))
        img = self.transforms(img)

        style_img = self.get_font_image(path, is_multi_font=True, is_random_pad=True, is_rote=is_rote, fontfile=random.choice(self.fontfile_paths), rote=rote)
        style_img = cv2.resize(style_img,(self.img_size, self.img_size))
        style_img = self.transforms(style_img)

        return img, style_img, label

    # Content：複数フォント＋手書き，位置自由，回転あり
    # Style：複数フォント, 位置固定，回転あり
    def step3(self, path, fontfile=None):
        char = path.split("/")[-3]
        label = self.char_list.index(char)
        label = torch.tensor(label)

        if int(random.uniform(0,1)+0.5):
            is_rote=False
            rote = None
        else:
            is_rote=True
            rote = random.randint(-15,15)

        if int(random.uniform(0,1)+0.25):
            is_random_pad = False
        else:
            is_random_pad = True

        if int(random.uniform(0,1)+0.5):
            img = self.get_font_image(path, is_multi_font=True, is_random_pad=False, is_rote=is_rote, fontfile=random.choice(self.fontfile_paths), rote=rote)
        else:
            img = self.get_handwrite_image(random.choice(self.char_path_dict[char]), is_random_pad=is_random_pad, is_rote=is_rote, rote=rote)
        img = cv2.resize(img,(self.img_size, self.img_size))
        img = self.transforms(img)

        style_img = self.get_font_image(path, is_multi_font=True, is_random_pad=False, is_rote=is_rote, fontfile=random.choice(self.fontfile_paths), rote=rote)
        style_img = cv2.resize(style_img,(self.img_size, self.img_size))
        style_img = self.transforms(style_img)

        return img, style_img, label

    # Content：複数フォント＋手書き，位置自由，回転あり
    # Style：複数フォント＋手書き, 位置固定，回転あり
    def step_end(self, path, fontfile=None):
        char = path.split("/")[-3]
        label = self.char_list.index(char)
        label = torch.tensor(label)

        if int(random.uniform(0,1)+0.5):
            is_rote=False
            rote = None
        else:
            is_rote=True
            rote = random.randint(-15,15)

        if int(random.uniform(0,1)+0.25):
            is_random_pad = False
        else:
            is_random_pad = True

        if int(random.uniform(0,1)+0.5):
            img = self.get_font_image(path, is_multi_font=True, is_random_pad=False, is_rote=is_rote, fontfile=random.choice(self.fontfile_paths), rote=rote)
        else:
            img = self.get_handwrite_image(path, is_random_pad=False, is_rote=is_rote, rote=rote)
        img = cv2.resize(img,(self.img_size, self.img_size))
        img = self.transforms(img)

        if int(random.uniform(0,1)+0.5):
            style_img = self.get_font_image(path, is_multi_font=True, is_random_pad=False, is_rote=is_rote, fontfile=random.choice(self.fontfile_paths), rote=rote)
        else:
            style_img = self.get_handwrite_image(random.choice(self.char_path_dict[char]), is_random_pad=False, is_rote=is_rote, rote=rote)
        style_img = cv2.resize(style_img,(self.img_size, self.img_size))
        style_img = self.transforms(style_img)

        return img, style_img, label

    def __getitem__(self, index):
        fontfile = self.indices[index]
        char = random.choice(self.char_list)
        path = random.choice(self.char_path_dict[char])

        if self.step == "step1":
            img, style_img, label = self.step1(path, fontfile)
        elif self.step == "step2":
            img, style_img, label = self.step2(path, fontfile)
        elif self.step == "step3":
            img, style_img, label = self.step3(path, fontfile)
        elif self.step == "step_end":
            img, style_img, label = self.step_end(path, fontfile)

        _, tokens_tensor,token_type_ids_tensor,attention_mask_tensor = self.get_tokens(char)
        return img, style_img, label, tokens_tensor,token_type_ids_tensor,attention_mask_tensor

    def __len__(self):
        return self.len

from torch.nn.utils.rnn import pad_sequence

#ミニバッチを取り出して長さを揃える関数
def img_collate_func(batch):
    imgs1,imgs2, labels, xs, ys, zs = [], [], [], [], [],[]
    for img1, img2, label, x,y,z in batch:
        imgs1.append(img1)
        imgs2.append(img2)
        labels.append(label)
        xs.append(torch.LongTensor(x))
        ys.append(torch.LongTensor(y))
        zs.append(torch.LongTensor(z))

    #データ長を揃える処理
    xs = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0.0)
    zs = pad_sequence(zs, batch_first=True, padding_value=0.0)
    imgs1 = torch.stack(imgs1)
    imgs2 = torch.stack(imgs2)
    labels = torch.stack(labels)
    return imgs1, imgs2, labels, xs, ys, zs