import torch
from torch.utils import data
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import cv2
import os
import albumentations as A
import random

from utils import crop_all_contours, adjust_square, known_pad, load_json

class Dataset(data.Dataset):
    def __init__(self, class_list, indices, config, phase):
        super().__init__()
        self.class_list = class_list
        self.img_size = config["img_size"]
        self.indices = indices
        self.len = len(self.indices)
        self.aug = config["aug"]
        self.phase = phase

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["mean"],
                std=config["std"]
            )
        ])

        elastictransform = A.ElasticTransform(alpha=1, 
                                            sigma=10, 
                                            alpha_affine=20, 
                                            interpolation=1, 
                                            border_mode=0, 
                                            value=None, 
                                            mask_value=None, 
                                            always_apply=False, 
                                            approximate=False, 
                                            same_dxdy=True, 
                                            p=0.5)
        resize = A.Resize(height=self.img_size, width=self.img_size, p=1)
        aug_list = [elastictransform, resize]
        self.compose = A.Compose(aug_list)

        self.kanji2element = load_json(config["element_json"])
        # ファイル全体をリストとして読み込み
        with open(config["vocabulary_list"]) as f:
            vocabulary_list = [s.strip() for s in f.readlines()]
        self.vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

    def path2img_etl(self, path):
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        h, w= img.shape[0], img.shape[1]
        s_h = 0.125
        s_w = 0.15
        img = img[round(h*s_h):h - round(h*s_h), round(w*s_w):w - round(w*s_w)]
        img = cv2.resize(img,(self.img_size, self.img_size))
        _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img) # 白黒反転

        img = crop_all_contours(img)
        img = cv2.bitwise_not(img) # 白黒反転
        img = adjust_square(img)
        img = known_pad(img)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img,(self.img_size, self.img_size))

        return img

    def get_tokens(self, char):
        tokens = []
        tokens_tensor = []

        elements = self.kanji2element[char]
        element = random.choice(elements)
        tokens = element.split(" ")

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_type_ids = [0]*len(tokens)
        attention_mask = [0 if t == "[PAD]" else 1 for t in tokens]
        tokens_tensor = [self.vocabulary_list.index(t) if t in self.vocabulary_list else 1 for t in tokens]

        token_type_ids_tensor = torch.tensor(token_type_ids)
        attention_mask_tensor = torch.tensor(attention_mask)
        tokens_tensor = torch.tensor(tokens_tensor)

        # inputs = {"input_ids":tokens_tensor, "token_type_ids":token_type_ids_tensor, "attention_mask":attention_mask_tensor}

        return tokens, tokens_tensor,token_type_ids_tensor,attention_mask_tensor

    def __getitem__(self, index):
        f = self.indices[index]
        char, path = f.split(" ")[0], f.split(" ")[1]

        img = self.path2img_etl(path)

        if self.phase =="train":
            img = cv2.bitwise_not(img)
            transformed = self.compose(image=img)
            img = transformed["image"]
            img = cv2.bitwise_not(img)

        _, tokens_tensor,token_type_ids_tensor,attention_mask_tensor = self.get_tokens(char)

        return self.transform(img), tokens_tensor,token_type_ids_tensor,attention_mask_tensor

    def __len__(self):
        return len(self.indices)

#ミニバッチを取り出して長さを揃える関数
def clip_collate_func(batch):
    imgs, xs, ys, zs = [], [], [],[]
    for img, x,y,z in batch:
        imgs.append(img)
        xs.append(torch.LongTensor(x))
        ys.append(torch.LongTensor(y))
        zs.append(torch.LongTensor(z))

    #データ長を揃える処理
    xs = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0.0)
    zs = pad_sequence(zs, batch_first=True, padding_value=0.0)
    imgs = torch.stack(imgs)
    return imgs, xs, ys, zs

class ImageDataset(data.Dataset):
    def __init__(self, class_list, indices, config, phase):
        super().__init__()
        self.class_list = class_list
        self.img_size = config["img_size"]
        self.indices = indices
        self.aug = config["aug"]
        self.phase = phase

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config["mean"],
                std=config["std"]
            )
        ])

        elastictransform = A.ElasticTransform(alpha=1, 
                                            sigma=10, 
                                            alpha_affine=20, 
                                            interpolation=1, 
                                            border_mode=0, 
                                            value=None, 
                                            mask_value=None, 
                                            always_apply=False, 
                                            approximate=False, 
                                            same_dxdy=True, 
                                            p=0.5)
        resize = A.Resize(height=self.img_size, width=self.img_size, p=1)
        aug_list = [elastictransform, resize]
        self.compose = A.Compose(aug_list)

    def path2img_etl(self, path):
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        h, w= img.shape[0], img.shape[1]
        s_h = 0.125
        s_w = 0.15
        img = img[round(h*s_h):h - round(h*s_h), round(w*s_w):w - round(w*s_w)]
        img = cv2.resize(img,(self.img_size, self.img_size))
        _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img) # 白黒反転

        img = crop_all_contours(img)
        img = cv2.bitwise_not(img) # 白黒反転
        img = adjust_square(img)
        img = known_pad(img)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img,(self.img_size, self.img_size))

        return img

    def __getitem__(self, index):
        f = self.indices[index]
        char, path = f.split(" ")[0], f.split(" ")[1]
        label = self.class_list.index(char)

        img = self.path2img_etl(path)

        if self.phase =="train":
            img = cv2.bitwise_not(img)
            transformed = self.compose(image=img)
            img = transformed["image"]
            img = cv2.bitwise_not(img)

        return self.transform(img), torch.tensor(label)

    def __len__(self):
        return len(self.indices)

class TextDataset(data.Dataset):
    def __init__(self, class_list, config, phase):
        super().__init__()
        self.class_list = class_list
        self.phase = phase

        self.kanji2element = load_json(config["element_json"])

        # 全てから選択
        if phase =="train":
            self.indices = self.class_list*10
        else:
            self.indices = self.class_list*1
        self.len = len(self.indices)
        # ファイル全体をリストとして読み込み
        with open(config["vocabulary_list"]) as f:
            vocabulary_list = [s.strip() for s in f.readlines()]
        self.vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

    def get_tokens(self, char):
        tokens = []
        tokens_tensor = []

        elements = self.kanji2element[char]
        element = random.choice(elements)
        tokens = element.split(" ")

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        token_type_ids = [0]*len(tokens)
        attention_mask = [0 if t == "[PAD]" else 1 for t in tokens]
        tokens_tensor = [self.vocabulary_list.index(t) if t in self.vocabulary_list else 1 for t in tokens]

        token_type_ids_tensor = torch.tensor(token_type_ids)
        attention_mask_tensor = torch.tensor(attention_mask)
        tokens_tensor = torch.tensor(tokens_tensor)

        # inputs = {"input_ids":tokens_tensor, "token_type_ids":token_type_ids_tensor, "attention_mask":attention_mask_tensor}

        return tokens, tokens_tensor,token_type_ids_tensor,attention_mask_tensor

    def __getitem__(self, index):
        char = self.indices[index]
        _, tokens_tensor,token_type_ids_tensor,attention_mask_tensor = self.get_tokens(char)

        label = self.class_list.index(char)
        label = torch.tensor(label)

        return tokens_tensor,token_type_ids_tensor,attention_mask_tensor, label

    def __len__(self):
        return self.len

#ミニバッチを取り出して長さを揃える関数
def text_collate_func(batch):
    xs, ys, zs, ls = [], [],[],[]
    for x,y,z,l in batch:
        xs.append(torch.LongTensor(x))
        ys.append(torch.LongTensor(y))
        zs.append(torch.LongTensor(z))
        ls.append(torch.LongTensor(l))

    #データ長を揃える処理
    xs = pad_sequence(xs, batch_first=True, padding_value=0.0)
    ys = pad_sequence(ys, batch_first=True, padding_value=0.0)
    zs = pad_sequence(zs, batch_first=True, padding_value=0.0)
    return xs, ys, zs, torch.tensor(ls)

    