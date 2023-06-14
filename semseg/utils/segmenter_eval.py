import torch
import numpy as np
import torch.distributed as dist

import os
import pickle as pkl
from pathlib import Path
import tempfile
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict

from timm.models.layers import trunc_normal_

"""
ImageNet classifcation accuracy
"""
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


def resize(im, smaller_size):
    h, w = im.shape[2:]
    if h < w:
        ratio = w / h
        h_res, w_res = smaller_size, ratio * smaller_size
    else:
        ratio = h / w
        h_res, w_res = ratio * smaller_size, smaller_size
    if min(h, w) < smaller_size:
        im_res = F.interpolate(im, (int(h_res), int(w_res)), mode="bilinear")
    else:
        im_res = im
    return im_res

def sliding_window(im, flip, window_size, window_stride):
    B, C, H, W = im.shape
    ws = window_size

    windows = {"crop": [], "anchors": []}
    h_anchors = torch.arange(0, H, window_stride)
    w_anchors = torch.arange(0, W, window_stride)
    h_anchors = [h.item() for h in h_anchors if h < H - ws] + [H - ws]
    w_anchors = [w.item() for w in w_anchors if w < W - ws] + [W - ws]
    for ha in h_anchors:
        for wa in w_anchors:
            window = im[:, :, ha : ha + ws, wa : wa + ws]
            windows["crop"].append(window)
            windows["anchors"].append((ha, wa))
    windows["flip"] = flip
    windows["shape"] = (H, W)
    return windows


def merge_windows(windows, window_size, ori_shape):
    ws = window_size
    im_windows = windows["seg_maps"]
    anchors = windows["anchors"]
    C = im_windows[0].shape[0]
    H, W = windows["shape"]
    flip = windows["flip"]

    logit = torch.zeros((C, H, W), device=im_windows.device)
    count = torch.zeros((1, H, W), device=im_windows.device)
    for window, (ha, wa) in zip(im_windows, anchors):
        logit[:, ha : ha + ws, wa : wa + ws] += window
        count[:, ha : ha + ws, wa : wa + ws] += 1
    logit = logit / count
    logit = F.interpolate(
        logit.unsqueeze(0),
        ori_shape,
        mode="bilinear",
    )[0]
    if flip:
        logit = torch.flip(logit, (2,))
    result = F.softmax(logit, 0)
    return result


def inference(
    model,
    ims,
    window_size,
    window_stride,
    batch_size,
):
    C = 151
    ori_shape = (512, 512)
    seg_map = torch.zeros((batch_size, C, ori_shape[0], ori_shape[1]), device='cuda')
    print(ims.size())
    # for im in ims:
    ims = ims.to('cuda')
    ims = resize(ims, window_size)
    flip = False
    windows = sliding_window(ims, flip, window_size, window_stride)
    crops = torch.stack(windows.pop("crop"))[:, 0]
    B = len(crops)
    WB = batch_size
    seg_maps = torch.zeros((B, C, window_size, window_size), device=ims.device)
    with torch.no_grad():
        for i in range(0, B, WB):
            seg_maps[i : i + WB] = model.forward(crops[i : i + WB])
    windows["seg_maps"] = seg_maps
    im_seg_map = merge_windows(windows, window_size, ori_shape)
    seg_map += im_seg_map
    seg_map /= len(ims)
    print(seg_map.size())
    return seg_map