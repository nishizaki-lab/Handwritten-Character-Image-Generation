import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
import os
import json
import time
from tqdm import tqdm
import cv2

from dataset import Dataset,img_collate_func
from model import Discriminator
from utils import (return_img, make_grid,
                    CLIP_fn, LPIPS, hinge_d_loss, vanilla_d_loss,
                    calc_mean_std, calculate_adaptive_weight, adopt_weight)

def main():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0)
    parser.add_argument("-mm", "--model_mood", type=str, default="AdaIN with CLIP") # 使用するモデルの種類
    parser.add_argument("--clip_weight", type=str, default=None) # 途中から生成モデルを学習する際に設定する
    parser.add_argument("--gen_weight", type=str, default=None) # 途中から生成モデルを学習する際に設定する
    parser.add_argument("--disc_weight", type=str, default=None) # 途中から判別モデルを学習する際に設定する

    parser.add_argument("--step_type", type=str, default="step1") # 使用するデータローダの訓練ステップ
    parser.add_argument("-size", "--img_size", type=int, default=256) # 画像サイズ(256しか試していない，128でも学習はできると思う)
    parser.add_argument("--use_font", type=str, default="../information/use_font.txt")
    parser.add_argument("--use_char", type=str, default="../information/hira_kana_kanji.txt")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("-batch", "--batch_size", type=int, default=8)
    parser.add_argument("-opt", "--optimizer", type=str, default="Adam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=4.5e-6)
    parser.add_argument('--aug', action='store_true')

    parser.add_argument("--element_json", type=str, default="../information/structure.json")
    parser.add_argument("--vocabulary_list", type=str, default="../information/vocabulary.txt")

    args = parser.parse_args()
    config = vars(args)
    config["command"] = " ".join(sys.argv)


    device = torch.device('cuda:%d'%(config["gpu"]) if torch.cuda.is_available() else 'cpu')

    save_folder = "{}_step-{}".format(config["model_mood"].replace(' ', '-'),config["step_type"])
    save_log = os.path.join("experiment", save_folder, 'log')
    save_model = os.path.join("experiment", save_folder, 'model')
    save_image = os.path.join("experiment", save_folder, 'image')
    os.makedirs(save_log, exist_ok=True)
    os.makedirs(save_model, exist_ok=True)
    os.makedirs(save_image, exist_ok=True)
    summary_writer = SummaryWriter(log_dir=save_log)
    with open(os.path.join("experiment", save_folder, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    if config["model_mood"] == "AdaIN with CLIP":
        from model import AdaIN_with_CLIP

        model = AdaIN_with_CLIP(
            device=device,
            config=config,
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            cond_drop_prob = 0.5
        ).to(device)
    
    elif config["model_mood"] == "AdaIN with MLP":
        from model import AdaIN_with_MLP

        model = AdaIN_with_MLP(
            config=config,
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            cond_drop_prob = 0.5
        ).to(device)
    
    elif config["model_mood"] == "AdaIN":
        from model import AdaIN

        model = AdaIN(
            dim = 64,
            dim_mults = (1, 2, 4, 8),
            cond_drop_prob = 0.5
        ).to(device)


    discriminator = Discriminator()

    if config["gen_weight"] is not None:
        gen_weight = torch.load(config["gen_weight"], map_location=device)
        model.load_state_dict(gen_weight)

    if config["disc_weight"] is not None:
        disc_weight = torch.load(config["disc_weight"], map_location=device)
        discriminator.load_state_dict(disc_weight)
   
    model = model.to(device)
    discriminator = discriminator.to(device)

    train_dataset = Dataset(config)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, collate_fn=img_collate_func)

    if config["optimizer"]=="SGD":
        gen_opt = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
        dis_opt = torch.optim.SGD(discriminator.parameters(), lr=config["learning_rate"])
    elif config["optimizer"]=="Adam":
        gen_opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.5, 0.9))
        dis_opt = torch.optim.Adam(discriminator.parameters(), lr=config["learning_rate"], betas=(0.5, 0.9))
    elif config["optimizer"]=="AdamW":
        gen_opt = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], betas=(0.5, 0.9))
        dis_opt = torch.optim.AdamW(discriminator.parameters(), lr=config["learning_rate"], betas=(0.5, 0.9))


    clip_loss_fn = CLIP_fn(config)
    implicit_loss_fn = nn.MSELoss()
    implicit_weight = 10

    perceptual_weight = 1.0
    perceptual_loss = LPIPS(device).eval()
    cond=None
    disc_conditional = False
    disc_factor_weight = 1.0
    discriminator_iter_start = 0
    disc_loss="hinge"
    if disc_loss == "hinge":
        disc_loss = hinge_d_loss
    elif disc_loss == "vanilla":
        disc_loss = vanilla_d_loss

    train_step = 0
    epochs = config["epochs"]
    for epoch in range(epochs):
        model.train()#学習モードに移行
        discriminator.train()
        training=True

        train_loss = 0
        s_time = time.time()
        for batch in tqdm(train_dataloader):
            image, s_image, l_label, l_tokens_tensor,l_token_type_ids_tensor,l_attention_mask_tensor = batch
            image, s_image, l_label, l_tokens_tensor,l_token_type_ids_tensor,l_attention_mask_tensor = image.to(device), s_image.to(device), l_label.to(device), l_tokens_tensor.to(device),l_token_type_ids_tensor.to(device),l_attention_mask_tensor.to(device)
            l_inputs = {"input_ids":l_tokens_tensor, "token_type_ids":l_token_type_ids_tensor, "attention_mask":l_attention_mask_tensor}

            if config["model_mood"] == "AdaIN with CLIP":
                quant, s_quant, c_quant = model.encode(image, s_image, l_inputs)
                dec = model.decode(quant, c_quant)
                l_s_quant = model.style_encoder(dec)
            elif config["model_mood"] == "AdaIN with MLP":
                quant, s_quant, c_quant = model.encode(image, s_image, l_label)
                dec = model.decode(quant, c_quant)
                l_s_quant = model.style_encoder(dec)
            elif config["model_mood"] == "AdaIN":
                quant, s_quant = model.encode(image, s_image)
                dec = model.decode(quant)
                l_s_quant = model.style_encoder(dec)

            gen_opt.zero_grad()

            explicit_loss = clip_loss_fn.compute_losses(dec, l_inputs).mean()

            s_feat_mean, s_feat_std = calc_mean_std(s_quant)
            l_s_feat_mean, l_s_feat_std = calc_mean_std(l_s_quant)
            implicit_loss = implicit_loss_fn(s_feat_mean, l_s_feat_mean) + implicit_loss_fn(s_feat_std, l_s_feat_std)

            if cond is None:
                assert not disc_conditional
                logits_fake = discriminator(dec.contiguous())
            else:
                assert disc_conditional
                logits_fake = discriminator(torch.cat((dec.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            rec_loss = torch.abs(s_image.contiguous() - dec.contiguous())
            if perceptual_weight > 0:
                p_loss = perceptual_loss(s_image.contiguous(), dec.contiguous())
                rec_loss = rec_loss + perceptual_weight * p_loss
            else:
                p_loss = torch.tensor([0.0])

            nll_loss = rec_loss
            nll_loss = torch.mean(nll_loss)
            try:
                d_weight = calculate_adaptive_weight(nll_loss, g_loss, last_layer=model.final_conv.weight)
            except RuntimeError:
                assert not training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(disc_factor_weight, train_step, threshold=discriminator_iter_start)

            total_g_step_loss = explicit_loss + implicit_weight*implicit_loss + nll_loss + d_weight * disc_factor * g_loss
            train_loss+=float(total_g_step_loss.to('cpu').detach().numpy().copy())

            if train_step % 1000 == 0:
                org_images = return_img(image)
                style_images = return_img(s_image)
                mix_images = return_img(dec)

                save_img = np.concatenate([org_images, style_images, mix_images], 0)
                save_img = make_grid(save_img, args.batch_size)
                cv2.imwrite(os.path.join(save_image, "train_ep{}-step{}.png".format(epoch, train_step)), save_img)

                if train_step % 5000 == 0:
                    summary_writer.add_image('gen_sample', save_img, train_step, dataformats='HWC')

            summary_writer.add_scalar("train_total_loss", total_g_step_loss.clone().detach(), train_step)
            summary_writer.add_scalar("train_nll_loss", nll_loss.clone().detach().mean(), train_step)
            summary_writer.add_scalar("train_d_weight", d_weight.clone().detach().mean(), train_step)
            summary_writer.add_scalar("train_disc_factor", disc_factor, train_step)
            summary_writer.add_scalar("train_g_loss", g_loss.detach().mean(), train_step)
            summary_writer.add_scalar("train_explicit_loss", explicit_loss.clone().detach(), train_step)
            summary_writer.add_scalar("train_implicit_loss", implicit_loss.clone().detach(), train_step)

            total_g_step_loss.backward()
            gen_opt.step()

            # discriminator update
            dis_opt.zero_grad()

            if cond is None:
                logits_real = discriminator(s_image.contiguous().detach())
                logits_fake = discriminator(dec.contiguous().detach())
            else:
                logits_real = discriminator(torch.cat((s_image.contiguous().detach(), cond), dim=1))
                logits_fake = discriminator(torch.cat((dec.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(disc_factor_weight, train_step, threshold=discriminator_iter_start)
            total_d_step_loss = disc_factor * disc_loss(logits_real, logits_fake)

            summary_writer.add_scalar("train_disc_loss", total_d_step_loss.clone().detach().mean(), train_step)
            summary_writer.add_scalar("train_logits_real", logits_real.clone().detach().mean(), train_step)
            summary_writer.add_scalar("train_logits_fake", logits_fake.clone().detach().mean(), train_step)

            total_d_step_loss.backward()
            dis_opt.step()

            if train_step % 25000 == 0:
                torch.save(model.state_dict(), os.path.join(save_model, "u-net_ae_model_step%s.pt"%(train_step)))
                torch.save(discriminator.state_dict(), os.path.join(save_model, "discriminator_ep_step%s.pt"%(train_step)))

            train_step+=1
        train_loss/=len(train_dataloader)

        print('Time for epoch {} is {} | Train Loss {}'.format(epoch+1, time.time() - s_time, train_loss))

    torch.save(model.state_dict(), os.path.join(save_model, "style_transfer_model.pt"))
    torch.save(discriminator.state_dict(), os.path.join(save_model, "discriminator.pt"))

if __name__ == "__main__":
    main()