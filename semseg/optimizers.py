from torch import nn
from torch.optim import AdamW, SGD
from semseg.layer_decay import add_params
import torch
from itertools import chain
from semseg.models.backbones.convnext_orig import *


def get_optimizer(model: nn.Module, optimizer: str, lr: float, weight_decay: float = 0.01):
    # params = []
 
    # lr=5e-4,
    # paramss = [
    #     {"params": group_weight(model.backbone, 0)},
    #     {"params": group_weight(model.get_decoder_params(), 1)}

    # ]
    paramss = group_weight(model)
    # add_params(paramss, model, lr, weight_decay)
    # print(paramss)
    # print(paramss)

    if optimizer == 'AdamW':
        print("AdamW")
        return AdamW(paramss, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    else:
        return SGD(paramss, lr, momentum=0.9, weight_decay=weight_decay)



def group_weight(model, idx = 0):
    group_decay = []
    group_no_decay = []
    bn_params = []
    bn_keys = []
    other_params = []
    # module= model.backbone
    # if idx == 0:
    modules= [model.backbone, model.decode_head, model.auxiliary_head]
    for j in modules:
        for name, param in j.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or "norm" in name: #or name in no_weight_decay_list
                group_no_decay.append(param)
                # group_no_decay.append(name.bias)
            else:
                group_decay.append(param)
    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    # module = model.decode_head
    # # else:
    # paramss = 0
    # for mod in module:
    #     for m in mod.modules():
    #         if isinstance(m, nn.Linear):
    #             group_decay.append(m.weight)
    #             if m.bias is not None:
    #                 group_no_decay.append(m.bias)
    #         elif isinstance(m, nn.Conv2d):
    #             group_decay.append(m.weight)
    #             if m.bias is not None:
    #                 group_no_decay.append(m.bias)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             if m.weight is not None:
    #                 group_no_decay.append(m.weight)
    #             if m.bias is not None:
    #                 group_no_decay.append(m.bias)
    #         elif isinstance(m, LayerNorm):
    #             if m.weight is not None:
    #                 group_no_decay.append(m.weight)
    #             if m.bias is not None:
    #                 group_no_decay.append(m.bias)
    #         elif isinstance(m, nn.Parameter):
    #             if m.weight is not None:
    #                 group_no_decay.append(m.weight)
    #             if m.bias is not None:
    #                 group_no_decay.append(m.bias)
    #     paramss+= len(list(mod.parameters()))
    # assert paramss == len(group_decay) + len(group_no_decay)

    # for m in module.modules():
    #     # print(name)
    #     idd+=1

  
        # elif isinstance(m, Block):
        #     for mm in m.modules():
        #         print(mm)
        #         idd+=1
        #         if mm.weight is not None:
        #             group_no_decay.append(mm.weight)
        #         if mm.bias is not None:
        #             group_no_decay.append(mm.bias)

    # print(idd, len(list(module.parameters())), len(group_decay), len(group_no_decay))
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

