import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from timm.models.registry import register_model
from timm.models.vision_transformer import (
    _create_vision_transformer,
    default_cfgs,
)

from semseg.models.backbones.vit_encoder import (
    VisionTransformer,
    resize_pos_embed,
)
from semseg.models.heads.segmenter_decoder import (
    DecoderLinear,
    MaskTransformer,
)
from semseg.utils.utils import *

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]


def resample_abs_pos_embed_nhwc(
    posemb,
    new_size,
    interpolation="bicubic",
    antialias=True,
    verbose=False,
):
    if new_size[0] == posemb.shape[-3] and new_size[1] == posemb.shape[-2]:
        return posemb

    orig_dtype = posemb.dtype
    posemb = posemb.float()
    # do the interpolation
    posemb = posemb.reshape(
        1, posemb.shape[-3], posemb.shape[-2], posemb.shape[-1]
    ).permute(0, 3, 1, 2)
    posemb = F.interpolate(posemb, size=new_size, mode=interpolation, antialias=antialias)
    posemb = posemb.permute(0, 2, 3, 1).to(orig_dtype)

    return posemb


def interpolate_pos_encoding(
    pos_embed: Tensor,
    new_img_size: int,
    old_img_size: int = 224,
    patch_size: int = 16,
) -> Tensor:
    """
    Interpolates the positional encoding of ViTs for new image resolution
    (adapted from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L174).
    It currently handles only square images.
    """
    N = pos_embed.shape[1]  # - 1
    npatch = new_img_size // patch_size  # ** 2
    w, h = new_img_size, new_img_size
    if npatch == N and w == h:
        print("Positional encoding not changed.")
        return pos_embed
    print(
        f"Interpolating positional encoding from {N} to {npatch} patch (size={patch_size})."
    )
    # class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed  # [:, 1:]
    dim = pos_embed.shape[-1]
    w0 = w // patch_size
    h0 = h // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(
            0, 3, 1, 2
        ),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed  # torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


def resize_rel_pos(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb_tok, posemb_grid = (
        posemb[:, :num_extra_tokens],
        posemb[0, num_extra_tokens:],
    )
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h, gs_w, -1).squeeze()
    # posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    # print(posemb_grid.size())
    return posemb_grid


def checkpoint_filter_fn_sam(state_dict):
    """Convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if "model" in state_dict:
        # For deit models
        state_dict = state_dict["model"]
    num_extra_tokens = 0  # 1 + ("dist_token" in state_dict.keys())
    patch_size = 63
    image_size = [512, 512]
    for k, v in state_dict.items():
        if k in [
            "blocks.2.attn.rel_pos_w",
            "blocks.2.attn.rel_pos_h",
            "blocks.5.attn.rel_pos_w",
            "blocks.5.attn.rel_pos_h",
            "blocks.8.attn.rel_pos_w",
            "blocks.8.attn.rel_pos_h",
            "blocks.11.attn.rel_pos_w",
            "blocks.11.attn.rel_pos_h",
        ]:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_rel_pos(
                v,
                None,
                (patch_size, patch_size + 1),
                num_extra_tokens,
            )
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(state_dict, model):
    """Convert patch embedding weight from manual patchify + linear proj to conv"""
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
    def __init__(self, encoder, decoder, n_cls, backbone):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = 16
        self.encoder = encoder
        self.decoder = decoder
        self.backbone = backbone

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

        x = self.encoder(im, pre_neck=True)

        # remove CLS/DIST tokens for decoding
        if "SAM" in self.backbone:
            num_extra_tokens = 0  # for sam model
        else:
            num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        # print(x.shape)
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


@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
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


def create_vit(model_cfg, pretrained=""):
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

    model = VisionTransformer(**model_cfg)

    if backbone == "vit_base_patch8_384":
        model = VisionTransformer(**model_cfg)
        # path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        # state_dict = torch.load(path, map_location="cpu")
        # filtered_dict = checkpoint_filter_fn(state_dict, model)
        # model.load_state_dict(filtered_dict, strict=True)
        pass
    else:
        ckpt = torch.load(pretrained, map_location="cpu")  # ['model']
        ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        for k in ["base_normalize.mean", "base_normalize.std"]:
            ckpt.pop(k, None)
        ckpt = {k.replace("base_", ""): v for k, v in ckpt.items()}
        # ckpt = load_parameters_from_disk(model, pretrained)
        # exit()
        # print(ckpt.keys())

        try:
            filtered_dict = checkpoint_filter_fn(ckpt, model)
            # print(filtered_dict.keys())
            model.load_state_dict(filtered_dict)
            print("standard loading")
        except:
            pass

    return model


def create_decoder(encoder, decoder_cfg, backbone):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    if "SAM" in backbone:
        decoder_cfg["d_encoder"] = 768
    else:
        decoder_cfg["d_encoder"] = 384

    decoder_cfg["patch_size"] = 16

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = decoder_cfg["d_encoder"]
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder


def create_segmenter(model_cfg, pretrained, backbone):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg, pretrained)
    decoder = create_decoder(encoder, decoder_cfg, backbone=backbone)
    model = SegMenter(encoder, decoder, n_cls=model_cfg["n_cls"], backbone=backbone)

    return model


def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path) as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location="cpu")
    # checkpoint = data["model"]
    ckpt = {k.replace("image_encoder.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(checkpoint, strict=True)

    return model, variant


class ImageNormalizer(nn.Module):
    def __init__(
        self, mean: Tuple[float, float, float], std: Tuple[float, float, float]
    ) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer("mean", torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


def normalize_model(
    model: nn.Module,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> nn.Module:
    layers = OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)])
    return nn.Sequential(layers)
