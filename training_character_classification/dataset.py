import torch
from torch.utils import data
from torchvision import transforms
import cv2
import os
import albumentations as A
import random

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
        img = cv2.imread(path)
        h, w= img.shape[0], img.shape[1]
        s = 0.1
        img = img[round(h*s):h - round(h*s), round(w*s):w - round(w*s)]
        img = cv2.resize(img,(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(img)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        return img

    def path2img_gen(self, path):
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        brank = 50
        img = cv2.bitwise_not(img) # 白黒反転
        img = cv2.copyMakeBorder(img, brank, brank, brank, brank, cv2.BORDER_CONSTANT)
        img = cv2.bitwise_not(img) # 白黒反転

        img = cv2.resize(img,(self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def path2img_diff_gen(self, path):
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
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