import torch
from torch.utils import data
from torchvision import transforms
import cv2
import os
import albumentations as A
import random

import utils as image_util

class Dataset(data.Dataset):
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

        img = image_util.crop_all_contours(img)
        img = cv2.bitwise_not(img) # 白黒反転
        img = image_util.adjust_square(img)
        img = image_util.known_pad(img)

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img,(self.img_size, self.img_size))

        return img

    def path2img_gen(self, path):
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(self.img_size, self.img_size))
        _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img) # 白黒反転
        img = image_util.crop_all_contours(img)
        img = cv2.bitwise_not(img) # 白黒反転
        img = image_util.adjust_square(img)
        img = image_util.known_pad(img)

        img = cv2.resize(img,(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def path2img_diff_gen(self, path):
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(self.img_size, self.img_size))
        _,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img) # 白黒反転
        img = image_util.crop_all_contours(img)
        img = cv2.bitwise_not(img) # 白黒反転
        img = image_util.adjust_square(img)
        img = image_util.known_pad(img)

        img = cv2.resize(img,(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def __getitem__(self, index):
        f = self.indices[index]
        char, path = f.split(" ")[0], f.split(" ")[1]
        label = self.class_list.index(char)

        if "ETL" in "/".join(path.split("/")[:-1]):
            img = self.path2img_etl(path)
        elif "style" in "/".join(path.split("/")[:-1]):
            img = self.path2img_gen(path)
        elif "diff" in "/".join(path.split("/")[:-1]):
            img = self.path2img_diff_gen(path)
        else:
            img = self.path2img_gen(path)

        if self.aug and self.phase =="train":
            img = cv2.bitwise_not(img)
            transformed = self.compose(image=img)
            img = transformed["image"]
            img = cv2.bitwise_not(img)

        return self.transform(img), torch.tensor(label)

    def __len__(self):
        return len(self.indices)