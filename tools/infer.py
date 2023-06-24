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
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.models.segmenter import create_segmenter
from semseg.optimizers import get_optimizer, create_optimizers, adjust_learning_rate
from semseg.utils.utils import fix_seeds, Logger, makedir, normalize_model, load_config_segmenter
from val import evaluate, Pgd_Attack
import semseg.utils.attacker as attacker
import semseg.datasets.transform_util as transform
from semseg.metrics import Metrics
import torchvision
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
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


def sizeof_fmt(num, suffix="Flops"):
    for unit in ["", "Ki", "Mi", "G", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.3f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Yi{suffix}"


def worse_case_eval(data_loder, l_output, n_cls=21, ignore_index=-1):
    """Compute worse case across 4-losses in SEA."""

    acc = 0
    n_ex = 0
    int_cls = torch.zeros(n_cls)
    union_cls = torch.zeros(n_cls)

    # l_output = []

    aa= [l_output] #, l_output2]
    final_acc_1 = None
    final_acc_2 = None

    class_wise_logits = torch.stack(l_output)
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
            pred = class_wise_logits[:, i*BS:i*BS+BS]

            acc_curr = pred == target
            intersection_all = pred == target

            for cl in range(n_cls):
                ind = target == cl
                ind = ind.expand(class_wise_logits.shape[0], -1, -1, -1)
                acc_curr_ = acc_curr  * ind
                acc_cls[:, :, cl] += acc_curr_.view(acc_curr.shape[0], acc_curr.shape[1], -1).float().sum((2))
                n_pxl_cls[:, :, cl] += ind.view(ind.shape[0], ind.shape[1], -1).float().sum(2)

                intersection_all_ = intersection_all * ind
                int_cls[:, :,  cl] = intersection_all_.view(intersection_all.shape[0], intersection_all.shape[1], -1).float().sum(2)
                union_cls[:, :,  cl] = (ind.view(ind.shape[0], ind.shape[1], -1).float().sum(-1) + (pred == cl).view(intersection_all.shape[0], intersection_all.shape[1], -1).float().sum(-1)
                              - intersection_all_.view(intersection_all.shape[0], intersection_all.shape[1], -1).float().sum(2))

            tenss = acc_cls.sum(2) / n_pxl_cls.sum(2)
            aaacc.append(tenss)
            ious.append(int_cls)
            unions.append(union_cls)
        if i + 1 == n_batches:
            print('enough batches seen')
            break
   
    final_acc_1 = (torch.cat(aaacc, dim=-1))
       

    ious = torch.cat(ious, dim=1)
    unions = torch.cat(unions, dim=1)
    wrs_i = []
    wrs_u = []

    print("loss-wise mious:", (ious.sum(1)/unions.sum(1)).mean(-1))
    indx = final_acc_1.min(0)[1]
    for i in range(indx.size(0)):
        wrs_i.append(ious[indx[i], i, :].unsqueeze(0))
        wrs_u.append(unions[indx[i], i, :].unsqueeze(0))

    worse_ious = torch.cat(wrs_i, dim=0)
    worse_unios = torch.cat(wrs_u, dim=0)
  

    worse_miou = (worse_ious.sum(0) / worse_unios.sum(0)).mean() 

    worse_1 = final_acc_1.min(0)[0].mean()
    at_w_sum1 = final_acc_1.mean(-1)
 
    print("SEA evaluated Acc", worse_1)
    print("SEA evaluated mIoU", worse_miou.item())

    pairs = []
    pair_lis =  [[0,1,2], [0,2,3], [0, 3], [0,1,3]] # used for table-6 in paper

    for p in pair_lis:
        pairs.append(final_acc_1[p].min(0)[0].mean().item())

    save_dict = {'worst_Acc_across_4_losses': worse_1,
                'worst_Acc_indiv': at_w_sum1, 
                'pair_wise_key': pair_idx,
                'pair_wise_Acc': pairs, 
                'final_matrix_Acc': final_acc_1,
                'worse_miou': worse_miou,
                'worse_inter_per_img': worse_ious,
                'worse_union_per_img': worse_unios}
    print(save_dict)

    return save_dict


def eval_performance(
    model, data_loder, n_batches=-1, n_cls=21, return_output=False, ignore_index=-1, return_preds=False, verbose=False):
    """Evaluate accuracy."""

    model.eval()
    acc = 0
    acc_cls = torch.zeros(n_cls)
    n_ex = 0
    n_pxl_cls = torch.zeros(n_cls)
    int_cls = torch.zeros(n_cls)
    union_cls = torch.zeros(n_cls)
    # if logger is None:
    #     logger = Logger(None)
    l_output = []

    for i, vals in enumerate(data_loder):
        if False:
            print(i)
        else:
            input, target = vals[0], vals[1]
            #print(input[0, 0, 0, :10])
            # print(input[0, 0, 0, :10], input.min(), input.max(),
            #     target.min(), target.max())
            input = input.cuda()

            with torch.no_grad():
                output = model(input)
            # l_output.append(output.cpu())


            pred = output.max(1)[1].cpu()
            l_output.append(pred)
            pred[target == ignore_index] = ignore_index
            acc_curr = pred == target

            # Compute correctly classified pixels for each class.
            for cl in range(n_cls):
                ind = target == cl
                acc_cls[cl] += acc_curr[ind].float().sum()
                n_pxl_cls[cl] += ind.float().sum()

            ind = n_pxl_cls > 0
            m_acc = (acc_cls[ind] / n_pxl_cls[ind]).mean()

            # Compute overall correctly classified pixels.

            a_acc = acc_cls.sum() / n_pxl_cls.sum()
            n_ex += input.shape[0]
            #print('step 2 done')

            # Compute intersection and union.
            intersection_all = pred == target
            for cl in range(n_cls):
                ind = target == cl
                int_cls[cl] += intersection_all[ind].float().sum()
                union_cls[cl] += (ind.float().sum() + (pred == cl).float().sum()
                                  - intersection_all[ind].float().sum())
            ind = union_cls > 0
            m_iou = (int_cls[ind] / union_cls[ind]).mean()

            if verbose:
                print(
                    f'batch={i} running mAcc={m_acc:.2%} running aAcc={a_acc.mean():.2%}',
                    f' running mIoU={m_iou:.2%}')


        if i + 1 == n_batches:
            print('enough batches seen')
            break


    l_output = torch.cat(l_output)
    stats = {
        'mAcc': m_acc.item(),
        'aAcc': a_acc.item(),
        'mIoU': m_iou.item()}

    return stats, l_output



def evaluate(val_loader, model, attack_fn, n_batches=-1, args=None):
    """Run attack on points."""

    model.eval()
    adv_loader = []

    for i, (input, target) in enumerate(val_loader):
        # print(input[0, 0, 0, :10])
        input = input.cuda()
        target = target.cuda()

        x_adv, _, acc = attack_fn(model, input.clone(), target)
        check_imgs(input, x_adv, norm=args.norm)
        if False:
            print(f'batch={i} avg. pixel acc={acc.mean():.2%}')

        adv_loader.append((x_adv.cpu(), target.cpu().clone()))
        if i + 1 == n_batches:
            break

    return adv_loader



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
        val_data, batch_size=test_cfg['BATCH_SIZE'], shuffle=False,
        num_workers=2, pin_memory=True, sampler=None, worker_init_fn =seed_worker, generator=g)

    return val_loader


class MaskClass(nn.Module):

    def __init__(self, ignore_index: int) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input: Tensor) -> Tensor:
        if self.ignore_index == 0:
            return input[:, 1:]
        else:
            return torch.cat(
                (input[:, :self.ignore_index],
                 input[:, self.ignore_index + 1:]), dim=1)


def mask_logits(model: nn.Module, ignore_index: int) -> nn.Module:
    # TODO: adapt for list of indices.
    layers = OrderedDict([
        ('model', model),
        ('mask', MaskClass(ignore_index))
    ])
    return nn.Sequential(layers)

los_pairs = ['mask-ce-avg', 'segpgd-loss', 'js-avg', 'mask-norm-corrlog-avg']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/pascalvoc_cvst_clean_5iter.yaml')
    parser.add_argument('--eps', type=float, default=4.)
    parser.add_argument('--n_iter', type=int, default=300) #donot change set to the standard SEA implementaiton
    parser.add_argument('--adversarial', action='store_true', help='adversarial eval?', default=True)
    parser.add_argument('--attack', type=str, default='segpgd-loss', help='pgd, cospgd-loss, ce-avg or mask-ce-avg, segpgd-loss, mask-norm-corrlog-avg, js-avg?')
    parser.add_argument('--attack_type', type=str, default='apgd-larg-eps', help='apgd or apgd-larg-eps?') #donot change set to the standard SEA implementaiton

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    dataset_cfg, model_cfg, test_cfg = cfg['DATASET'], cfg['MODEL'], cfg['EVAL']

    # model = eval(model_cfg['NAME'])(test_cfg['BACKBONE'], test_cfg['N_CLS'],None)

    if model_cfg['NAME'] != 'UperNetForSemanticSegmentation':
        model_cfg1, dataset_cfg1 = load_config_segmenter(backbone=model_cfg['BACKBONE'])
        model = create_segmenter(model_cfg1, model_cfg['PRETRAINED'])
    else:
        model = eval(model_cfg['NAME'])(test_cfg['BACKBONE'], test_cfg['N_CLS'],None)

    ckpt = torch.load(test_cfg['MODEL_PATH'], map_location='cpu')

    model.load_state_dict(ckpt)

    model = model.to('cuda')

    model.eval()

    val_data_loader = get_data(dataset_cfg, test_cfg)

    console.print(f"Model > [yellow1]{cfg['MODEL']['NAME']} {test_cfg['BACKBONE']}[/yellow1]")
    console.print(f"Dataset > [yellow1]{test_cfg['NAME']}[/yellow1]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)

    loss_wise_logits = []
    
    save_argmax_of_log = False #set to true for saving argmax of image-wise logits after adversarial attack
    verbose=False #print all intermediate step computations?

    #check the clean performance of the model
    clean_stats, _ = eval_performance(model, val_data_loader, n_batches=-1, n_cls=test_cfg['N_CLS']-1, ignore_index=-1, verbose=verbose)
    args.norm = 'Linf'

    if args.norm == 'Linf' and args.eps >= 1.:
        args.eps /= 255.

    for ite, ls in enumerate(los_pairs):
        console.print(f"Loss > [yellow1]{ls}[/yellow1]" + f" Epsilon > [yellow1]{args.eps:.4f}[/yellow1]")
        args.attack = ls
        if args.adversarial:
            n_batches = -1 # set to -1 for full validation set


            args.n_iter = 300
            attack_fn =  partial(
                attacker.apgd_largereps,
                norm=args.norm,
                eps=args.eps,
                n_iter=args.n_iter,
                n_restarts=1, #args.n_restarts,
                use_rs=True,
                loss=args.attack if args.attack else 'ce-avg',
                verbose=verbose,
                track_loss='norm-corrlog-avg' if args.attack == 'mask-norm-corrlog-avg' else 'ce-avg',
                log_path=None,
                early_stop=True)
            adv_loader = evaluate(val_data_loader, model, attack_fn, n_batches, args)

        if args.adversarial:
            adv_stats, l_outs = eval_performance(model, adv_loader, n_batches, n_cls=dataset_cfg['N_CLS'], ignore_index=-1, verbose=verbose)
            if save_argmax_of_log:
                torch.save(l_outs, cfg['SAVE_DIR'] + f"/argmax_logits_model_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_loss_{ls}_eps_{args.eps}.pt")
            # adv = torch.cat([x for x, y in adv_loader], dim=0).cpu()
            # data_dict = {'adv': adv}
            # print(data_dict['adv'].shape)
            loss_wise_logits.append(l_outs)

        #Write the stats for the individual loss to a text file
        if verbose:
            with open(cfg['SAVE_DIR'] + f"/loss_wise_stats_model_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_loss_{ls}_eps_{args.eps}.txt", 'a+') as f:
                if ite == 0:
                    f.write(f"{cfg['MODEL']['NAME']} - {test_cfg['BACKBONE']}\n")
                    f.write(f"{str(test_cfg['MODEL_PATH'])}\n")
                if args.adversarial:
                    f.write(f"----- Linf radius: {args.eps:.4f} ------")
                    f.write(f"Attack: {args.attack_type} {args.attack} \t \t Iterations: {args.n_iter} \t alpha: 0.01 \n")   
                    f.write(f"Adversarial results: {adv_stats}\n") 
                f.write("\n")

    sea_stats = worse_case_eval(val_data_loader, loss_wise_logits)
    torch.save(sea_stats, cfg['SAVE_DIR'] + f"/SEA_stats_{model_cfg['NAME']}_{model_cfg['BACKBONE']}_loss_{ls}_eps_{args.eps}.pt")
    console.rule(f"[cyan]Segmentation Ensemble Attack complete")
