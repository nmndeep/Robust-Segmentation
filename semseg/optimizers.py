from torch import nn
from torch.optim import AdamW, SGD
from semseg.layer_decay import add_params
import torch
from itertools import chain
from semseg.models.backbones.convnext_orig import *


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01, dataset='pascalaug', backbone='ConvNext-T_CVST'):
    # params = []
 
    # lr=5e-4,
    # paramss = [
    #     {"params": group_weight(model.backbone, 0)},
    #     {"params": group_weight(model.get_decoder_params(), 1)}

    # ]
    # if dataset == 'ade20k':
    #     #hard-coded from convnext-orig configs
    #     if 'ConvNeXt-T_CVST' in backbone:
    #         ll = 6
    #     elif 'ConvNeXt-S_CVST' in backbone:
    #         ll = 12
    #     paramss = []
    #     paramss.extend(group_weight(model.decode_head))# add_params(paramss, model, lr, weight_decay)
    #     paramss.extend(group_weight(model.auxiliary_head))
    #     paramss = add_params(paramss, model.backbone, lr, weight_decay, ll)

    # else:
    paramss = group_weight(model)
    # print(len(paramss))
    # print(paramss[-1])
    # print(paramss)

    if optimizer == 'AdamW':
        return AdamW(paramss, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(paramss, lr, momentum=0.9, weight_decay=weight_decay)



def group_weight(model):
    group_decay = []
    group_no_decay = []
    bn_params = []
    bn_keys = []
    other_params = []
    # module= model.backbone
    # if idx == 0:
    # modules= [model.backbone, model.decode_head, model.auxiliary_head]
    # for j in model:
    num_req_grad = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            num_req_grad+=1
            continue
        if param.ndim <= 1 or "norm" in name: #or name in no_weight_decay_list
            group_no_decay.append(param)
            # group_no_decay.append(name.bias)
        else:
            group_decay.append(param)
    assert len(list(model.parameters())) - num_req_grad == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(model, lr, wd, rank):

    optimizer_encoder = AdamW(
        group_weight(model.backbone, 0),
        lr=lr*0.1,
        betas=(0.9, 0.999),
        weight_decay=wd)
    optimizer_decoder = AdamW(
        group_weight(model.get_decoder_params(), 1),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=wd*0.95)
    return (optimizer_encoder, optimizer_decoder)


def adjust_learning_rate(optimizers, cur_iter, lr, max_iter, pow=0.9):
    scale_running_lr = ((1. - float(cur_iter) / max_iter) ** pow)
    running_lr_encoder = lr * scale_running_lr
    running_lr_decoder = lr * 0.5 * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = running_lr_decoder


# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import json


def get_num_layer_layer_wise(var_name, num_max_layer=12):
    
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.downsample_layers"):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    elif var_name.startswith("backbone.stages"):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = num_max_layer
        return layer_id
    else:
        return num_max_layer + 1


def get_num_layer_stage_wise(var_name, num_max_layer):
    if var_name in ("backbone.cls_token", "backbone.mask_token", "backbone.pos_embed"):
        return 0
    elif var_name.startswith("backbone.downsample_layers"):
        return 0
    elif var_name.startswith("backbone.stages"):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return num_max_layer - 1
        

def add_params(params, module, base_lr, base_wd, num_layers=6):
    """Add all parameters of module to the params list.
    The parameters of the given module will be added to the list of param
    groups, with specific rules defined by paramwise_cfg.
    Args:
        params (list[dict]): A list of param groups, it will be modified
            in place.
        module (nn.Module): The module to be added.
        prefix (str): The prefix of the module
        is_dcn_module (int|float|None): If the current module is a
            submodule of DCN, `is_dcn_module` will be passed to
            control conv_offset layer's learning rate. Defaults to None.
    """
    parameter_groups = {}
    # print(self.paramwise_cfg)
    num_layers = num_layers + 2
    decay_rate = 0.9
    decay_type = "stage_wise"
    print("Build LearningRateDecayOptimizerConstructor %s %f - %d" % (decay_type, decay_rate, num_layers))
    weight_decay = base_wd

    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in ('pos_embed', 'cls_token'):
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if decay_type == "layer_wise":
            layer_id = get_num_layer_layer_wise(name, self.paramwise_cfg.get('num_layers'))
        elif decay_type == "stage_wise":
            layer_id = get_num_layer_stage_wise(name, num_layers)
            
        group_name = "layer_%d_%s" % (layer_id, group_name)

        if group_name not in parameter_groups:
            scale = decay_rate ** (num_layers - layer_id - 1)

            parameter_groups[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
                "param_names": [], 
                "lr_scale": scale, 
                "group_name": group_name, 
                "lr": scale * base_lr, 
            }

        parameter_groups[group_name]["params"].append(param)
        parameter_groups[group_name]["param_names"].append(name)

    # print(parameter_groups)
    # rank, _ = get_dist_info()
    # if rank == 0:
    #     to_display = {}
    #     for key in parameter_groups:
    #         to_display[key] = {
    #             "param_names": parameter_groups[key]["param_names"], 
    #             "lr_scale": parameter_groups[key]["lr_scale"], 
    #             "lr": parameter_groups[key]["lr"], 
    #             "weight_decay": parameter_groups[key]["weight_decay"], 
    #         }
    #     print("Param groups = %s" % json.dumps(to_display, indent=2))
    params.extend(parameter_groups.values())
    
    return params

