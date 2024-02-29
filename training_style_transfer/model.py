import torch
from torch import nn

from functools import partial
from einops import rearrange, reduce, repeat

from model_utils import (ResnetBlock, Residual, PreNorm,
                        LinearAttention, Attention,
                        Downsample, Upsample,prob_mask_like,
                        ActNorm, CLIPDualEncoderModel)

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class Trained_CLIP_TextEncoder(nn.Module):
    def __init__(self, device, model_path, char_size, vocabulary_list, trainable: bool = False) -> None:
        super().__init__()

        vocabulary_list = ["[PAD]", "[UNK]","[CLS]","[SEP]"] + vocabulary_list

        model = CLIPDualEncoderModel(device=device, 
                                    char_size=char_size,
                                    image_encoder_alias="resnet50", 
                                    vocabulary_size=len(vocabulary_list), 
                                    image_embedding_dims=2048)
        weight = torch.load(model_path)
        model.load_state_dict(weight)

        for param in model.parameters():
            param.requires_grad = trainable

        self.encoder = model.text_encoder
        self.text_projection = model.text_projection

    def forward(self, input_ids):
        text_features = self.encoder(input_ids)
        text_embeddings = self.text_projection(text_features)

        return text_embeddings

##########################
#### AdaIN with CLIP #####
##########################
class AdaIN_with_CLIP(nn.Module):
    def __init__(
        self,
        device,
        config,
        dim,
        cond_drop_prob = 0.5,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        style_encoder_weight_path=None
    ):
        super().__init__()

        self.channels = channels
        input_channels = channels
        classes_dim = dim * 4
        self.cond_drop_prob = cond_drop_prob

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # AdaIN layers
        self.adain = adaptive_instance_normalization

        # class embeddings
        with open(config["use_char"]) as f:
            use_list = [s.strip() for s in f.readlines()]

        with open(config["vocabulary_list"]) as f:
            vocabulary_list = [s.strip() for s in f.readlines()]

        self.textencoder = Trained_CLIP_TextEncoder(device=device,
                                                    model_path=config["clip_weight"],
                                                    char_size=len(use_list),
                                                    vocabulary_list=vocabulary_list)
        self.null_classes_emb = nn.Parameter(torch.randn(256))
        self.post_textencoder = nn.Sequential(
            nn.Linear(256, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding = 3)

        # determine content encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)

        # determine style encoder
        self.autoencoder = AutoEncoder(
            dim = dim,
            dim_mults = dim_mults
        )
        if style_encoder_weight_path is not None:
            weight = torch.load(style_encoder_weight_path)
            self.autoencoder.load_state_dict(weight)
        self.style_encoder = self.autoencoder.encoder

        # determine decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))
        self.out_dim = channels
        self.final_res_block = block_klass(dim, dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def content_encoder(self, x, c):
        x = self.init_conv(x)

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, None, c)

            x = block2(x, None, c)
            x = attn(x)

            x = downsample(x)

        x = self.mid_block1(x, None, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, None, c)

        return x

    def decoder(self, x, c):
        for block1, block2, attn, upsample in self.ups:
            x = block1(x, None, c)

            x = block2(x, None, c)
            x = attn(x)

            x = upsample(x)

        x = self.final_res_block(x, None, c)
        return self.final_conv(x)

    def post_classes_mlp(self, x, classes):
        batch, device = x.shape[0], x.device
        classes_emb = self.textencoder(classes)

        if self.cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - self.cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.post_textencoder(classes_emb)

        return c

    def encode(self, x, style_image, classes):
        c = self.post_classes_mlp(x, classes)

        x  = self.content_encoder(x, c)
        s_x = self.style_encoder(style_image)

        x = self.adain(x, s_x)

        return x, s_x, c

    def decode(self, x, c):
        x = self.decoder(x, c)

        return x

    def forward(self,x,style_image,classes):
        x, _, c = self.encode(x, style_image, classes)
        x = self.decode(x, c)

        return x

#########################
#### AdaIN with MLP #####
#########################
class AdaIN_with_MLP(nn.Module):
    def __init__(
        self,
        config,
        dim,
        cond_drop_prob = 0.5,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        style_encoder_weight_path=None
    ):
        super().__init__()

        self.channels = channels
        input_channels = channels
        classes_dim = dim * 4
        self.cond_drop_prob = cond_drop_prob

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # AdaIN layers
        self.adain = adaptive_instance_normalization

        # class embeddings
        with open(config["use_char"]) as f:
            use_list = [s.strip() for s in f.readlines()]

        self.classes_emb = nn.Embedding(len(use_list), dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))

        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding = 3)

        # determine content encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)

        # determine style encoder
        self.autoencoder = AutoEncoder(
            dim = dim,
            dim_mults = dim_mults
        )
        if style_encoder_weight_path is not None:
            weight = torch.load(style_encoder_weight_path)
            self.autoencoder.load_state_dict(weight)
        self.style_encoder = self.autoencoder.encoder

        # determine decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))
        self.out_dim = channels
        self.final_res_block = block_klass(dim, dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def content_encoder(self, x, c):
        x = self.init_conv(x)

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, None, c)

            x = block2(x, None, c)
            x = attn(x)

            x = downsample(x)

        x = self.mid_block1(x, None, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, None, c)

        return x

    def decoder(self, x, c):
        for block1, block2, attn, upsample in self.ups:
            x = block1(x, None, c)

            x = block2(x, None, c)
            x = attn(x)

            x = upsample(x)

        x = self.final_res_block(x, None, c)

        return self.final_conv(x)

    def post_classes_mlp(self, x, classes):
        batch, device = x.shape[0], x.device

        # derive condition, with condition dropout for classifier free guidance

        classes_emb = self.classes_emb(classes)

        if self.cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - self.cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )

        c = self.classes_mlp(classes_emb)

        return c

    def encode(self, x, style_image, classes):
        c = self.post_classes_mlp(x, classes)

        x  = self.content_encoder(x, c)
        s_x = self.style_encoder(style_image)

        x = self.adain(x, s_x)

        return x, s_x, c

    def decode(self, x, c):
        x = self.decoder(x, c)

        return x

    def forward(self, x, style_image, classes):
        x, _, c = self.encode(x, style_image, classes)
        x = self.decode(x, c)

        return x

################
#### AdaIN #####
################
class AdaIN(nn.Module):
    def __init__(
        self,
        dim,
        cond_drop_prob = 0.5,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8,
        style_encoder_weight_path=None
    ):
        super().__init__()

        self.channels = channels
        input_channels = channels
        classes_dim = dim * 4
        self.cond_drop_prob = cond_drop_prob

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        # AdaIN layers
        self.adain = adaptive_instance_normalization

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding = 3)

        # determine content encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)

        # determine style encoder
        self.autoencoder = AutoEncoder(
            dim = dim,
            dim_mults = dim_mults
        )
        if style_encoder_weight_path is not None:
            weight = torch.load(style_encoder_weight_path)
            self.autoencoder.load_state_dict(weight)
        self.style_encoder = self.autoencoder.encoder

        # determine decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))
        self.out_dim = channels
        self.final_res_block = block_klass(dim, dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def content_encoder(self, x):
        x = self.init_conv(x)

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, None, None)

            x = block2(x, None, None)
            x = attn(x)

            x = downsample(x)

        x = self.mid_block1(x, None, None)
        x = self.mid_attn(x)
        x = self.mid_block2(x, None, None)

        return x

    def decoder(self, x):
        for block1, block2, attn, upsample in self.ups:
            x = block1(x, None, None)

            x = block2(x, None, None)
            x = attn(x)

            x = upsample(x)

        x = self.final_res_block(x, None, None)

        return self.final_conv(x)

    def encode(self, x, style_image):
        x  = self.content_encoder(x)
        s_x = self.style_encoder(style_image)

        x = self.adain(x, s_x)

        return x, s_x

    def decode(self, x):
        x = self.decoder(x)

        return x

    def forward(self, x, style_image):
        x, _ = self.encode(x, style_image)
        x = self.decode(x)

        return x

######################
#### AutoEncoder #####
######################
class AutoEncoder(nn.Module):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        resnet_block_groups = 8
    ):
        super().__init__()

        self.channels = channels
        input_channels = channels
        classes_dim = dim * 4

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        self.init_conv = nn.Conv2d(input_channels, dim, 7, padding = 3)

        # determine encoder
        self.downs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, classes_emb_dim = classes_dim)

        # determine decoder
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                block_klass(dim_out, dim_out, classes_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))
        self.out_dim = channels
        self.final_res_block = block_klass(dim, dim, classes_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def encoder(self, x):
        x = self.init_conv(x)
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, None, None)
            x = block2(x, None, None)
            x = attn(x)

            x = downsample(x)

        x = self.mid_block1(x, None, None)
        x = self.mid_attn(x)
        x = self.mid_block2(x, None, None)
        return x

    def decoder(self, x):
        for block1, block2, attn, upsample in self.ups:
            x = block1(x, None, None)
            x = block2(x, None, None)
            x = attn(x)

            x = upsample(x)

        x = self.final_res_block(x, None, None)
        return self.final_conv(x)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

########################
#### Discriminator #####
########################
class Discriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(Discriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)
