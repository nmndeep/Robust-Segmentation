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


EPS = 0.0471 #
ITERR = 2 #, #5
strr = [
f"apgd_mask-ce-avg_{ITERR}iter_rob_mod_{EPS}_n_it_100_pascalvoc_ConvNeXt-T_CVST_ROB.pt", 
f"apgd_segpgd-loss_{ITERR}iter_rob_mod_{EPS}_n_it_100_pascalvoc_ConvNeXt-T_CVST_ROB.pt",
f"apgd_js-avg_{ITERR}iter_rob_mod_{EPS}_n_it_100_pascalvoc_ConvNeXt-T_CVST_ROB.pt",
f"apgd_mask-norm-corrlog-avg_{ITERR}iter_rob_mod_{EPS}_n_it_100_pascalvoc_ConvNeXt-T_CVST_ROB.pt"
]

losses_lis = ['mask-ce-avg','segpgd-loss', 'js-avg','mask-norm-corrlog-avg']


# exit()
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
    for i in range(len(strr)):
        l_output.append(torch.load(BASE_DIR + strr[i]))
    class_wise_logits = torch.stack(l_output)
    # print(class_wise_logits.size())
    # final_acc = []
    aaacc = []
    for i, vals in enumerate(data_loder):
       

        if False:
            print(i)
        else:
            _, target = vals[0], vals[1]
            BS = target.shape[0]
            acc_cls = torch.zeros(class_wise_logits.shape[0], BS, n_cls)
            n_pxl_cls = torch.zeros(class_wise_logits.shape[0], BS, n_cls)

            output = class_wise_logits[:, i*BS:i*BS+BS]
            pred = output.max(2)[1].cpu()
            pred[:, (target == ignore_index)] = ignore_index
            acc_curr = pred == target

            for cl in range(n_cls):
                ind = target == cl
                acc_curr = acc_curr * ind
                acc_cls[:, :, cl] += acc_curr.view(acc_curr.shape[0], acc_curr.shape[1], -1).float().sum((2))
                n_pxl_cls[:, :, cl] += ind.view(ind.shape[0], -1).float().sum(1)
            ind = n_pxl_cls > 0
       
            tenss = acc_cls.sum(2) / n_pxl_cls.sum(2)
            # print(tenss.size())
            aaacc.append(tenss)

        if i + 1 == n_batches:
            print('enough batches seen')
            break
    final_acc = torch.cat(aaacc, dim=-1)
    at_w_sum = final_acc.min(0)[1].unique(return_counts=True)[1]
    final_dict = dict((el, at_w_sum[i].item()) for i, el in enumerate(losses_lis))
    print(final_dict)
 
    with open(BASE_DIR + f"WORST_CASE_{strr[0][-62:-3]}.txt", 'a+') as f:
        f.write(f"LOSS-wise : {final_dict}\n")
    # print(stats)

    # return stats

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
        val_data, batch_size=32, shuffle=False,
        num_workers=1, pin_memory=True, sampler=None, worker_init_fn =seed_worker, generator=g)

    return val_loader

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