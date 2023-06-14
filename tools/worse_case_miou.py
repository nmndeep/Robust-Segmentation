import torch
import argparse
import yaml
import math, random
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
import torch.utils.data as data
from torchvision import transforms
from rich.console import Console
import numpy as np
from matplotlib import pyplot as plt
import cv2
from collections import OrderedDict

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tools.val import Pgd_Attack, clean_accuracy
from PIL import Image, ImageDraw, ImageFont
import gc
from autoattack.other_utils import check_imgs
import torch.nn as nn
from functools import partial
from semseg.utils.visualize import generate_palette
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer, create_optimizers, adjust_learning_rate
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, Logger, makedir, normalize_model
from val import evaluate, Pgd_Attack
import semseg.utils.attacker as attacker
import semseg.datasets.transform_util as transform
from semseg.metrics import Metrics
import torchvision
# from mmcv.utils import Config
# from mmcv.runner import get_dist_info
from semseg.datasets.dataset_wrappers import *
console = Console()
SEED = 225
random.seed(SEED)
np.random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


BASE_DIR = '/data/naman_deep_singh/model_zoo/seg_models/test_results/output_logits_new/'
# apgd_mask-ce-avg_5iter_rob_mod_0.0471_n_it_100_pascalvoc_ConvNeXt-T_CVST_ROB_SD_220.pt


DATAS = 'pascalvoc' # 'pascalvoc'
EPS = 0.0627 #0.0157, 0.0314, 0.0471, 0.0627
ITERR = "S_mod" #, #"S_mod"
ATTACK = 'apgd-larg-eps'
n_it = 300

strr = [
f"{ATTACK}_mask-ce-avg_{ITERR}_rob_mod_{EPS}_n_it_{n_it}_{DATAS}_ConvNeXt-S_CVST_ROB_SD_225.pt", 
f"{ATTACK}_segpgd-loss_{ITERR}_rob_mod_{EPS}_n_it_{n_it}_{DATAS}_ConvNeXt-S_CVST_ROB_SD_225.pt",
f"{ATTACK}_js-avg_{ITERR}_rob_mod_{EPS}_n_it_{n_it}_{DATAS}_ConvNeXt-S_CVST_ROB_SD_225.pt", 
f"{ATTACK}_mask-norm-corrlog-avg_{ITERR}_rob_mod_{EPS}_n_it_{n_it}_{DATAS}_ConvNeXt-S_CVST_ROB_SD_225.pt"
]

losses_lis = ['mask-ce-avg','segpgd-loss', 'js-avg','mask-norm-corrlog-avg']

pair_idx = [['ma-ce+seg+js'], ['ma-ce+js+ma-nm'], ['ma-ce+ma-mn'], ['ma-ce+seg+ma-nm']]



def clean_accuracy(
    data_loder, n_batches=-1, n_cls=21, return_output=False, ignore_index=-1, return_preds=False):
    """Evaluate accuracy."""

    model.eval()
    acc = 0
    n_ex = 0
    int_cls = torch.zeros(n_cls)
    union_cls = torch.zeros(n_cls)
    # if logger is None:
    #     logger = Logger(None)
    l_output = []
    l_output2 = []

    if ITERR == '5iter_mod':
        fold = '5iter_rob_model'
    elif ITERR == '2iter':
        fold = '2iter_rob_model'
    elif ITERR == 'S_mod':
        fold = 'S_model'
    else:
        fold = 'clean_model_out'
    # fold = '5iter_ade'
    # fold = '2iter_ade'
    print("before loading pt files")
    for i in range(len(strr)):
        # tens1 = torch.load(BASE_DIR + f"{fold}/preds/" + strr[i])
        # print(tens1.size())
        # tenss = tens1.max(1)[1]
        # torch.save(tenss, BASE_DIR + f"{fold}/preds/" + strr[i][:-3]+"_MAX.pt")
        l_output.append(torch.load(BASE_DIR + f"{fold}/preds/" + strr[i]))
        # l_output2.append(torch.load(BASE_DIR + f"{fold}/" +strr[i][:-3]+ "_SD_220.pt"))
        # print(l_output[-1].size())
    # exit()
    print("after loading pt files")
    aa= [l_output] #, l_output2]
    final_acc_1 = None
    final_acc_2 = None
    for j in range(1):
        class_wise_logits = torch.stack(aa[j])
        # class_wise_logits2 = torch.stack(l_output2)
        # class_wise_logits = 
        # print(class_wise_logits.size())

        aaacc = []
        ious = []
        unions = []
        for i, vals in enumerate(data_loder):
           
            if False:
                print(i)
            else:
                _, target = vals[0], vals[1]
                BS = target.shape[0]
                acc_cls = torch.zeros(class_wise_logits.shape[0], BS, n_cls)
                n_pxl_cls = torch.zeros(class_wise_logits.shape[0], BS, n_cls)
                # pointwise tensors for worst-case miou
                int_cls = torch.zeros(class_wise_logits.shape[0], BS, n_cls)
                union_cls = torch.zeros(class_wise_logits.shape[0], BS, n_cls)
                counts = torch.zeros_like(acc_cls)

                # pred = class_wise_logits[:, i*BS:i*BS+BS].cpu()
                output = class_wise_logits[:, i*BS:i*BS+BS].cpu()
                pred = output.max(2)[1].cpu()
                # pred[:, (target == ignore_index)] = ignore_index
                acc_curr = pred == target
                intersection_all = pred == target
                # print("intersection-all", intersection_all.size())
                for cl in range(n_cls):
                    ind = target == cl
                    ind = ind.expand(class_wise_logits.shape[0], -1, -1, -1)
                    acc_curr_ = acc_curr  * ind
                    acc_cls[:, :, cl] += acc_curr_.view(acc_curr.shape[0], acc_curr.shape[1], -1).float().sum((2))
                    n_pxl_cls[:, :, cl] += ind.view(ind.shape[0], ind.shape[1], -1).float().sum(2)

                    intersection_all_ = intersection_all * ind
                    int_cls[:, :,  cl] += intersection_all_.view(intersection_all.shape[0], intersection_all.shape[1], -1).float().sum(-1)
                    union_cls[:, :,  cl] += n_pxl_cls[:, :, cl] + (pred == cl).view(intersection_all.shape[0], intersection_all.shape[1], -1).float().sum(-1) - int_cls[:, :, cl]

                tenss = acc_cls.sum(2) / n_pxl_cls.sum(2)
                print(tenss.size())
                ind = (n_pxl_cls > 0)
                # print(ind.size())
                aaacc.append(tenss)
                ious.append(int_cls)
                unions.append(union_cls)

            if i + 1 == n_batches:
                print('enough batches seen')
                break
       

        if j == 0:
            final_acc_1 = (torch.cat(aaacc, dim=-1))
        else:
            final_acc_2 = (torch.cat(aaacc, dim=-1))
    #select worse loss depending on aAcc
    # print(ious[0].size())
    print(final_acc_1.min(0)[1].size())
    ious = torch.cat(ious, dim=1)
    unions = torch.cat(unions, dim=1)
    wrs_i = []
    wrs_u = []
    # print(ious.size())
    print("loss-wise mious:", (ious.sum(1)/unions.sum(1)).mean(-1))
    indx = final_acc_1.min(0)[1]
    for i in range(indx.size(0)):
        wrs_i.append(ious[indx[i], i, :].unsqueeze(0))
        wrs_u.append(unions[indx[i], i, :].unsqueeze(0))

    worse_ious = torch.cat(wrs_i, dim=0)
    worse_unios = torch.cat(wrs_u, dim=0)
  
    print(worse_ious.size(), worse_unios.size())
    # sum over classes and compute miou
    # exit()
    worse_miou = (worse_ious.sum(0) / worse_unios.sum(0)).mean() 

    worse_1 = final_acc_1.min(0)[0].mean()
    at_w_sum1 = final_acc_1.mean(-1)
 
    print("Loss-wise: Acc", at_w_sum1)
    print("Worse Acc across run", worse_1)
    print("Worse miou across run", worse_miou.item())
    pairs = []
    pair_lis =  [[0,1,2], [0,2,3], [0, 3], [0,1,3]]
    for p in pair_lis:
        print("Pair-wise worse for", p)
        print(final_acc_1[p].min(0)[0].mean())
        pairs.append(final_acc_1[p].min(0)[0].mean().item())

    save_dict = {'worst_Acc_across_4_losses': worse_1,
                'worst_Acc_per_loss': at_w_sum1, 
                'pair_wise_key': pair_idx,
                'pair_wise_Acc': pairs, 
                'final_matrix_Acc': final_acc_1,
                'worse_miou_across_4_losses': worse_miou,
                'miou_per_loss': (ious.sum(1)/unions.sum(1)).mean(-1),
                'inter_per_img_per_loss': ious,
                'union_per_img_per_loss': unions}

    torch.save(save_dict, BASE_DIR + f"/{fold}/worse_cases/WORST_CASE_{ATTACK}_{EPS}_{ITERR}" + strr[0][strr[0].find("300")+3:])


def get_data(dataset_cfg, test_cfg):

    if str(test_cfg['NAME']) == 'pascalvoc':
        data_dir = '../VOCdevkit/'
        val_data = get_segmentation_dataset(test_cfg['NAME'],
            root=dataset_cfg['ROOT'],
            split='val',
            transform=torchvision.transforms.ToTensor(),
            base_size=512,
            crop_size=(473, 473))

    elif str(test_cfg['NAME']) == 'pascalaug':
        val_data = get_segmentation_dataset(test_cfg['NAME'],
            root=dataset_cfg['ROOT'],
            split='val',
            transform=torchvision.transforms.ToTensor(),
            base_size=512,
            crop_size=(473, 473))

    elif str(test_cfg['NAME']).lower() == 'ade20k':
        val_data = get_segmentation_dataset(test_cfg['NAME'],
            root=dataset_cfg['ROOT'],
            split='val',
            transform=torchvision.transforms.ToTensor(),
            base_size=520,
            crop_size=(512, 512))
    else:
        raise ValueError(f'Unknown dataset.')


    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=48, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None, worker_init_fn =seed_worker, generator=g, drop_last=True)

    return val_loader

worse_comp = ['WORST_CASE_5iter_rob_mod_0.0471_n_it_100_pascalvoc_ConvNeXt-T_CVST_ROB_over_2_seeds.pt']

# 'WORST_CASE_S_mod_rob_mod_0.0471_n_it_100_pascalvoc_ConvNeXt-S_CVST_ROB.pt']
# , 'WORST_CASE_5iter_rob_mod_0.0627_n_it_100_pascalvoc_ConvNeXt-T_CVST_ROB.pt']
# 'WORST_CASE_5iter_rob_mod_0.0157_n_it_100_pascalvoc_ConvNeXt-T_CVST_ROB.pt',
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/pascalvoc_cvst_clean.yaml')
    parser.add_argument('--eps', type=float, default=4.)
    parser.add_argument('--store-data', action='store_true', help='PGD data?', default=False)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--adversarial', action='store_true', help='adversarial eval?', default=False)
    parser.add_argument('--attack', type=str, default='segpgd-loss', help='pgd, cospgd-loss, ce-avg or mask-ce-avg, segpgd-loss, mask-norm-corrlog-avg, js-avg?')
    parser.add_argument('--attack_type', type=str, default='apgd', help='pgd or pgd?')

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    dataset_cfg, model_cfg, test_cfg = cfg['DATASET'], cfg['MODEL'], cfg['EVAL']

    if False:
        out_str = "worst_case_" + worse_comp[0][10:16] + worse_comp[0][-37:-3]
        eps_ = ['4', '12']
        liss = []
        indices = [[1, 4, 5], [1, 5], [1, 2, 5], [2, 4, 5], [1, 2]]
        final_dict = {}
        indx_str = ['mask-ce+js+mask-norm', 'mask-ce+mask-norm', 'mask-ce+segpgd+mask-norm', 'segpgd+js+mask-norm', 'mask-ce+segpgd']
# iterating through the elements of list
        for i in eps_:
            final_dict[i] = None
        import json 
        if ITERR == '5iter':
            fold = '5iter_rob_model'
        elif ITERR == '2iter':
            fold = '2iter_rob_model'
        else:
            fold = 'S_model'
    
        for i in range(len(worse_comp)):
            vall = torch.load(BASE_DIR + f"{fold}/logs/{worse_comp[i]}")
            final_dict[eps_[i]] = {}
            print(vall['loss_wise_worse'])
            exit()
            final_dict[eps_[i]]['worst_all'] = vall['worst_case_across_losses'].item()
            for j in range(len(indices)):
                final_acc_ = vall['final_matrix'][indices[j]]
                final_dict[eps_[i]][indx_str[j]] = final_acc_.min(0)[0].mean().item() 


        json.dump(final_dict, open(BASE_DIR + f"/{fold}/{out_str}.txt", 'w'))
        # with open(BASE_DIR + f"worst_case_numbers/{out_str}.txt", 'a+') as f:
        #         pickle.dump(final_dict, f)

    else:
        model = eval(model_cfg['NAME'])(test_cfg['BACKBONE'], test_cfg['N_CLS'],None)
        model.load_state_dict(torch.load(test_cfg['MODEL_PATH'], map_location='cpu'))
        model_norm = False
        if model_norm:
            print('Add normalization layer.')   
            model = normalize_model(model, IN_MEAN, IN_STD)
        # model = mask_logits(model, 0)
        model = model.to('cuda')
        val_data_loader = get_data(dataset_cfg, test_cfg)
        clean_stats = clean_accuracy(val_data_loader, n_batches=-1, n_cls=test_cfg['N_CLS'], ignore_index=-1)