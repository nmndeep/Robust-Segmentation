from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer
# from timm.models.vision_transformer import VisionTransformer

from semseg.models.backbones.vit_encoder import VisionTransformer, ConvBlock, resize_pos_embed
from semseg.models.heads.segmenter_decoder import DecoderLinear, MaskTransformer
from semseg.utils.utils import normalize_model
from timm.models import create_model
from semseg.utils.utils import *

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]

def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 1 + ("dist_token" in state_dict.keys())
    patch_size = model.patch_size
    image_size = model.patch_embed.image_size
    for k, v in state_dict.items():
        if k == "pos_embed" and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v,
                None,
                (image_size[0] // patch_size, image_size[1] // patch_size),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict


def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im.size(2), im.size(3)
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y



class SegMenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = 16
        self.encoder = encoder
        self.decoder = decoder

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)



# import os
# import glob

# npz_files = {
#     'S_16_224_imagenet1k': '/data/naman_deep_singh/model_zoo/clean/vit_s_clean.npz',}
# # Create strategy and run server
# def load_parameters_from_disk(model, loc):
#     for name, filename in npz_files.items():
        
#         # Load Jax weights
#         npz = np.load(filename)

#         # Load PyTorch model
#         # model = pytorch_pretrained_vit.ViT(name=name, pretrained=False)

#         # Convert weights
#         new_state_dict = convert(npz, model.state_dict())

#         # Load into model and test
#         model.load_state_dict(new_state_dict)
#         print(f'Checking: {name}')
#         check_model(model, name)

#         # Save weights
#         # new_filename = f'weights/{name}.pth'
#         torch.save(new_state_dict, f"vit_{name}.pth", _use_new_zipfile_serialization=False)
#         print(f"Converted {filename} and saved to {new_filename}")

    


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model

def create_vit(model_cfg, pretrained=''):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    model = VisionTransformer(**model_cfg)
    # model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    # model = _create_vision_transformer('vit_small_patch16_224', pretrained=True, **dict(model_args))
    # model.patch_embed.proj = ConvBlock(48, end_siz=8)
    # model.load_pretrained(checkpoint_path=pretrained)
    for name, child in model.named_children():
        for namm, pamm in child.named_parameters():
            print(name + ' ' + namm)
                    
    #     print('Add normalization layer.')   
    #     model = normalize_model(model, IN_MEAN, IN_STD)
    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    else:

        ckpt = torch.load(pretrained, map_location='cpu') #['model']
        ckpt = {k.replace('model.', ''): v for k, v in ckpt.items()}
        ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        for k in ['base_normalize.mean', 'base_normalize.std']:
            ckpt.pop(k, None)
        ckpt = {k.replace('base_', ''): v for k, v in ckpt.items()}
        # ckpt = load_parameters_from_disk(model, pretrained)
        # exit()
        # print(ckpt.keys())

        try:
            filtered_dict = checkpoint_filter_fn(ckpt, model)
            print(filtered_dict.keys())
            model.load_state_dict(filtered_dict)
            print('standard loading')
        except:
            pass
        #     try:
        #         # ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
        #         filtered_dict = checkpoint_filter_fn(ckpt, model)
        #         model.load_state_dict(filtered_dict)
        #         print('loaded from clean model')
        #     except:
        #         ckpt = {k.replace('base_model.', ''): v for k, v in ckpt.items()}
        #         filtered_dict = checkpoint_filter_fn(ckpt, model)
        #         # ckpt = {f'base_model.{k}': v for k, v in ckpt.items()}
        #         model.load_state_dict(filtered_dict)
        #         print('loaded')

    return model

def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = 384
    decoder_cfg["patch_size"] = 16

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = 384
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_cfg, pretrained):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg, pretrained)
    decoder = create_decoder(encoder, decoder_cfg)
    model = SegMenter(encoder, decoder, n_cls=model_cfg["n_cls"])

    return model


def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant