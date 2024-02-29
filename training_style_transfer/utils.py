import json

def load_json(filename):
    with open(filename) as f:
        d = json.load(f)
    return d

def get_etl_train_val_data():
    train_list = []
    val_list = []

    path = "../data_paths/ETL_train/all_paths.json"
    with open(path) as f:
        train_d = json.load(f)

    path = "../data_paths/ETL_val/all_paths.json"
    with open(path) as f:
        val_d = json.load(f)

    return train_d, val_d

import numpy as np

def make_grid(imgs, nrow, padding=0):
    """Numpy配列の複数枚の画像を、1枚の画像にタイルします

    Arguments:
        imgs {np.ndarray} -- 複数枚の画像からなるテンソル
        nrow {int} -- 1行あたりにタイルする枚数

    Keyword Arguments:
        padding {int} -- グリッドの間隔 (default: {0})

    Returns:
        [np.ndarray] -- 3階テンソル。1枚の画像
    """
    assert imgs.ndim == 4 and nrow > 0
    batch, height, width, ch = imgs.shape
    n = nrow * (batch // nrow + np.sign(batch % nrow))
    ncol = n // nrow
    pad = np.zeros((n - batch, height, width, ch), imgs.dtype)
    x = np.concatenate([imgs, pad], axis=0)
    # border padding if required
    if padding > 0:
        x = np.pad(x, ((0, 0), (0, padding), (0, padding), (0, 0)),
                   "constant", constant_values=(0, 0)) # 下と右だけにpaddingを入れる
        height += padding
        width += padding
    x = x.reshape(ncol, nrow, height, width, ch)
    x = x.transpose([0, 2, 1, 3, 4])  # (ncol, height, nrow, width, ch)
    x = x.reshape(height * ncol, width * nrow, ch)
    if padding > 0:
        x = x[:(height * ncol - padding),:(width * nrow - padding),:] # 右端と下端のpaddingを削除
    return x

import torch

def return_img(image):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    image = image.to("cpu")

    x = image.mul(torch.FloatTensor(IMAGENET_STD).view(3, 1, 1))
    x = x.add(torch.FloatTensor(IMAGENET_MEAN).view(3, 1, 1)).detach().numpy()
    x = np.transpose(x, (0, 2, 3, 1))

    image = np.clip(x, 0, 1)
    image = (image*255).astype(np.uint8)

    return image

import albumentations as A
from PIL import Image, ImageDraw, ImageFont,ImageOps
import cv2
import random

img_size = 256
rotate = A.Rotate(always_apply=False, 
                p=1, 
                limit=(-15, 15), 
                interpolation=0, 
                border_mode=0, 
                value=(0, 0, 0), 
                mask_value=None, 
                method='largest_box', 
                crop_border=False)

resize = A.Resize(height=img_size, width=img_size, p=1)
aug_list = [rotate, resize]
albumentations_aug = A.Compose(aug_list)


up_dot = ['"', "'", "^", "°","゜","「"]
under_dot = [',','，','.',',','。','、',"_", "」"]
center = ["・",'・',"一", "-", "ー", "―", "~","=", "丶"]
small = ["ぁ", "ぃ", "ぅ","ぇ","ぉ","っ","ゃ","ゅ","ょ","ゎ",
        "ァ", "ィ", "ゥ","ェ","ォ","ッ","ャ","ュ","ョ","ヮ","ㇲ","ㇻ"]
def char2font(char, fontfile, is_aug=False, is_pad=False):
    try:
        org_char = 224
        img = Image.new('RGB', ((org_char+30)*len(char), org_char+30), 'white')
        draw = ImageDraw.Draw(img)
        draw.font = ImageFont.truetype(fontfile, org_char)
        draw.text([0, -int(org_char*0.05)], char, (0, 0, 0))
        img = ImageOps.invert(img)

        img = img.crop(img.getbbox())
        img = ImageOps.invert(img)
        img = np.array(img)

        if np.sum(cv2.bitwise_not(img))==0 or np.sum(cv2.bitwise_not(img))>=255*img.shape[0]*img.shape[1]*3*0.95 and char!="一":
            return None
        else:
            if char not in up_dot + under_dot+ center+ small:

                # if int(random.uniform(0,1)+0.5):
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #     img = ocrodeg.random_blotches(img, 3e-4, 1e-4)
                #     img = (img*255).astype(np.uint8)
                #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # h, w , _ = img.shape
                # back_img = np.full((org_char, w, 3), 255, dtype=np.uint8)
                # if h < org_char:
                #     ymin = random.randint(0, org_char-h)
                # else:
                #     img = img[:org_char,:,:]
                #     ymin = 0

                # back_img[ymin:ymin+img.shape[0],:img.shape[1],:] = img
                # img = back_img

                img = adjust_square(img)
                if is_pad:
                    img = random_pad(img)

                # if int(random.uniform(0,1)+0.5):
                #     img = self.add_background_image(img)

                if is_aug:
                    img = cv2.bitwise_not(img)
                    transformed = albumentations_aug(image=img)
                    img = transformed["image"]
                    img = cv2.bitwise_not(img)

                return img
            else:
                if char in small:
                    img = adjust_square(img)
                    img = adjust_square(img, int((org_char - img.shape[1])/2))
                elif char in up_dot:
                    img = adjust_square(img)

                    if img.shape[0]>int(org_char/2):
                        img = cv2.resize(img, (int(org_char/2), int(org_char/2)))
                        back_img = np.full((int(org_char/2), org_char, 3), 255, dtype=np.uint8)
                        back_img[:int(org_char/2),:int(org_char/2),:] = img
                        img = back_img
                    else:
                        back_img = np.full((int(org_char/2), org_char, 3), 255, dtype=np.uint8)
                        up_margin = random.randint(0, int(org_char/2)-img.shape[0])
                        left_margin = random.randint(0, org_char-img.shape[1])

                        back_img[up_margin:up_margin+img.shape[0],left_margin:left_margin+img.shape[1],:] = img
                        img = back_img

                    if char !="「":
                        back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)
                        back_img[:int(org_char/2),:,:] = img
                        img = back_img
                    else:
                        back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)
                        up_margin = random.randint(0, int(org_char/1.5)-img.shape[0])
                        back_img[up_margin:up_margin+img.shape[0],:,:] = img
                        img = back_img

                elif char in up_dot:
                    img = adjust_square(img)

                    if img.shape[0]>int(org_char/2):
                        img = cv2.resize(img, (int(org_char/2), int(org_char/2)))
                        back_img = np.full((int(org_char/2), org_char, 3), 255, dtype=np.uint8)
                        back_img[:,int(org_char/2):,:] = img
                        img = back_img
                    else:
                        back_img = np.full((int(org_char/2), org_char, 3), 255, dtype=np.uint8)
                        up_margin = random.randint(0, int(org_char/2)-img.shape[0])
                        left_margin = random.randint(0, org_char-img.shape[1])

                        back_img[up_margin:up_margin+img.shape[0],left_margin:left_margin+img.shape[1],:] = img
                        img = back_img

                    if char !="」":
                        back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)
                        back_img[int(org_char/2):,:,:] = img
                        img = back_img
                    else:
                        back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)
                        up_margin = random.randint(int(org_char/1.5)-img.shape[0], int(org_char-img.shape[0]))
                        back_img[up_margin:up_margin+img.shape[0],:,:] = img
                        img = back_img

                elif char in center:
                    img = adjust_square(img)

                    if img.shape[0]>int(org_char/1.5):
                        img = cv2.resize(img, (int(org_char/2), int(org_char/2)))

                    back_img = np.full((org_char, org_char, 3), 255, dtype=np.uint8)

                    up_margin = random.randint(int(org_char/1.5)-img.shape[0], int(org_char)-img.shape[0])
                    up_margin = int(up_margin/2)
                    left_margin = random.randint(int(org_char/1.5)-img.shape[1], int(org_char)-img.shape[1])
                    left_margin = int(left_margin/2)

                    back_img[up_margin:up_margin+img.shape[0],left_margin:left_margin+img.shape[1],:] = img
                    img = back_img

                return img
    except:
        return None
    

def path2handimg(path, is_aug = False):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    # ETL9
    img = img[round(h*0.1):h - round(h*0.1), round(w*0.17):w - round(w*0.15)]
    # img = img[round(h*0.1):h - round(h*0.1), round(w*0.1):w - round(w*0.1)]
    img = cv2.resize(img,(img_size, img_size))
    ret3,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img = cv2.bitwise_not(img) # 白黒反転
    brank = 80
    img = cv2.copyMakeBorder(img, brank, brank, brank, brank, cv2.BORDER_CONSTANT)
    img = cv2.bitwise_not(img) # 白黒反転
    img = crop_img(img, brank)

    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if is_aug:
        img = cv2.bitwise_not(img)
        transformed = albumentations_aug(image=img)
        img = transformed["image"]
        img = cv2.bitwise_not(img)
    return img


def adjust_square(img, blank=0):
    h, w, _ = img.shape
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

def random_pad(img, max_pad=40, pad_value=0):
    h, w, _ = img.shape
    img = cv2.bitwise_not(img)
    up_pad = random.randint(10, max_pad)
    down_pad = random.randint(10, max_pad)
    left_pad = random.randint(10, max_pad)
    right_pad = random.randint(10, max_pad)
    # 正方形にする
    img = cv2.copyMakeBorder(img, up_pad, down_pad, right_pad, left_pad, cv2.BORDER_CONSTANT, pad_value)
    img = cv2.bitwise_not(img)
    return img

def known_pad(img, pad=20, pad_value=0):
    h, w, _ = img.shape
    img = cv2.bitwise_not(img)
    up_pad, down_pad, left_pad, right_pad = pad, pad, pad, pad
    # 正方形にする
    img = cv2.copyMakeBorder(img, up_pad, down_pad, right_pad, left_pad, cv2.BORDER_CONSTANT, pad_value)
    img = cv2.bitwise_not(img)
    return img

def crop_img(img, brank):
    h_exit_pix = []
    w_exit_pix = []
    space = 10
    for i in range(img_size+brank*2):
        h_pix_sum = np.sum(img[i])
        w_pix_sum = np.sum(img[:, i])
        if h_pix_sum !=255*(img_size+brank*2):
            h_exit_pix.append(i)

        if w_pix_sum !=255*(img_size+brank*2):
            w_exit_pix.append(i)

    h_s = h_exit_pix[0]
    h_e = h_exit_pix[-1]
    w_s = w_exit_pix[0]
    w_e = w_exit_pix[-1]

    h = h_e - h_s
    w = w_e - w_s
    # print(h, w)

    if h > w:
        add_w = round((h - w)/2)
        w_s = w_s - add_w 
        w_e = w_e + add_w
    elif h < w:
        add_h = round((w - h)/2)
        h_s = h_s - add_h 
        h_e = h_e + add_h

    if h_s < space:
        h_s = space

    if h_e > img_size+brank*2 -space:
        h_e = img_size+brank*2 -space

    if w_s < space:
        w_s = space

    if w_e > img_size+brank*2 -space:
        w_e = img_size+brank*2 -space

    img = img[h_s-space: h_e+space, w_s-space: w_e+space]
    img = cv2.resize(img,(img_size, img_size))

    return img

def rote_img(img, angle=45):
    h, w, _ = img.shape
    center = (int(w/2), int(h/2))
    trans = cv2.getRotationMatrix2D(center, angle , 1)
    img = cv2.bitwise_not(img)
    img = cv2.warpAffine(img, trans, (w,h))
    img = cv2.bitwise_not(img)
    return img

import torch
import torch.nn.functional as F
from torch import nn
from model import CLIPDualEncoderModel

class CLIP_fn(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        with open(config["use_char"]) as f:
            use_list = [s.strip() for s in f.readlines()]

        with open(config["vocabulary_list"]) as f:
            vocabulary_list = [s.strip() for s in f.readlines()]
        vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

        self.model = CLIPDualEncoderModel(device="cpu", 
                                    char_size=len(use_list),
                                    image_encoder_alias="resnet50", 
                                    vocabulary_size=len(vocabulary_list), 
                                    image_embedding_dims=2048)
        weight = torch.load(config["clip_weight"], map_location="cpu")
        self.model.load_state_dict(weight)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(config["gpu"])
        self.model.eval()

        self.temperature = 1.0
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def compute_losses(self, image, inputs):
        image_embeddings, text_embeddings = self.model(image,inputs)

        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = (-targets.T * self.log_softmax(logits.T)).sum(1)
        texts_loss = (-targets * self.log_softmax(logits)).sum(1)
        return (images_loss + texts_loss) / 2.0

def calculate_norm(r_style, l_style):
    # MSE = nn.MSELoss()
    return torch.mean(torch.cdist(r_style, l_style, p=2))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import namedtuple
import os
import hashlib
import requests
import numpy as np
from tqdm import tqdm

def calculate_adaptive_weight(nll_loss, g_loss, last_layer=None, discriminator_weight=1.0):
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * discriminator_weight
    return d_weight

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, device, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer(device)
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False).to(device)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout).to(device)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout).to(device)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout).to(device)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout).to(device)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout).to(device)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val

class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)

class ScalingLayer(nn.Module):
    def __init__(self, device):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None].to(device))
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None].to(device))

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out

URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}

MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}

def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)

def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std