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

IN_MEAN = [0.485, 0.456, 0.406]
IN_STD = [0.229, 0.224, 0.225]

def clean_accuracy(
    model, data_loder, n_batches=-1, n_cls=21, return_output=False, ignore_index=-1, return_preds=False):
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
    #print('Using {n_cls} classes and ignore')

    metrics = Metrics(n_cls, -1, 'cpu')

    for i, vals in enumerate(data_loder):
        if False:
            print(i)
        else:
            input, target = vals[0], vals[1]
            #print(input[0, 0, 0, :10])
            print(input[0, 0, 0, :10], input.min(), input.max(),
                target.min(), target.max())
            input = input.cuda()

            with torch.no_grad():
                output = model(input)
            if return_preds:
                l_output.append(output.cpu())
            #print('fp done')
            #metrics.update(output.cpu(), target)

            pred = output.max(1)[1].cpu()
            pred[target == ignore_index] = ignore_index
            acc_curr = pred == target
            #print('step 1 done')

            # Compute correctly classified pixels for each class.
            for cl in range(n_cls):
                ind = target == cl
                acc_cls[cl] += acc_curr[ind].float().sum()
                n_pxl_cls[cl] += ind.float().sum()
            #print(acc_cls, n_pxl_cls)
            ind = n_pxl_cls > 0
            m_acc = (acc_cls[ind] / n_pxl_cls[ind]).mean()

            # Compute overall correctly classified pixels.
            #acc_curr = acc_curr.float().view(input.shape[0], -1).mean(-1)
            #acc += acc_curr.sum()
            a_acc = acc_cls.sum() / n_pxl_cls.sum()
            n_ex += input.shape[0]
            #print('step 2 done')

            # Compute intersection and union.
            intersection_all = pred == target
            #pred[target == 0] = 0
            for cl in range(n_cls):
                ind = target == cl
                int_cls[cl] += intersection_all[ind].float().sum()
                union_cls[cl] += (ind.float().sum() + (pred == cl).float().sum()
                                  - intersection_all[ind].float().sum())
            ind = union_cls > 0
            #ind[0] = False
            m_iou = (int_cls[ind] / union_cls[ind]).mean()

            print(
                f'batch={i} running mAcc={m_acc:.2%} running aAcc={a_acc.mean():.2%}',
                f' running mIoU={m_iou:.2%}')

        #print(metrics.compute_iou()[1], metrics.compute_pixel_acc()[1])

        if i + 1 == n_batches:
            print('enough batches seen')
            break

    # logger.log(f'mAcc={m_acc:.2%} aAcc={a_acc:.2%} mIoU={m_iou:.2%} ({n_ex} images)')
    #print(acc_cls / n_pxl_cls)
    #print(acc_cls.sum() / n_pxl_cls.sum())
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
        print(input[0, 0, 0, :10])
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
        val_data, batch_size=test_cfg['BATCH_SIZE'], shuffle=True,
        num_workers=1, pin_memory=True, sampler=None)

    return val_loader



# alpha, num_iters
attack_setting = {'pgd': (0.01, 40), 'segpgd': (0.01, 40),
                    'cospgd': (0.15, 40),
                    'maskpgd': (0.15, 40)}



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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k_convnext_vena.yaml')
    parser.add_argument('--eps', type=float, default=16.)
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

    # clean_accuracy(model, dataloader)
    # exit()
    console.print(f"Model > [yellow1]{cfg['MODEL']['NAME']} {test_cfg['BACKBONE']}[/yellow1]")
    console.print(f"Dataset > [yellow1]{test_cfg['NAME']}[/yellow1]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)

    preds = []
    lblss = []

    clean_stats, _ = clean_accuracy(model, val_data_loader, n_batches=-1, n_cls=test_cfg['N_CLS'], ignore_index=-1)
    # print(clean_stats)
    # exit()
    for ite, ls in enumerate(['ce-avg', 'mask-ce-avg', 'segpgd-loss', 'cospgd-loss', 'js-avg', 'mask-norm-corrlog-avg']):
        args.attack = ls #'segpgd-loss' 
        if args.adversarial:
            strr = f'adversarial_loss_comparison_{args.attack_type}'
        else:
            strr = 'clean'
        # args.eps = ls
        if args.adversarial:
            n_batches = -1
            # norm = 'Linf'
            args.norm = 'Linf'

            if args.norm == 'Linf' and args.eps >= 1.:
                args.eps /= 255.

            attack_pgd = Pgd_Attack(epsilon=args.eps, alpha=1e-2, num_iter=100, los=args.attack) if args.attack_type == 'pgd' else None

            attack_fn = partial(
                attacker.apgd_restarts,
                norm=args.norm,
                eps=args.eps,
                n_iter=args.n_iter,
                n_restarts=1,
                use_rs=True,
                loss=args.attack if args.attack else 'ce-avg',
                verbose=True,
                track_loss='norm-corrlog-avg' if args.attack == 'mask-norm-corrlog-avg' else 'ce-avg',    
                log_path=None,
                ) if args.attack_type == 'apgd' else partial(attack_pgd.adv_attack)

            adv_loader = evaluate(val_data_loader, model, attack_fn, n_batches, args)
        

        if args.adversarial:
            adv_stats, l_outs = clean_accuracy(model, adv_loader, n_batches, n_cls=dataset_cfg['N_CLS'], ignore_index=-1)
            torch.save(l_outs, cfg['SAVE_DIR'] + "/test_results/output_logits/" + f"{args.attack_type}_S_model_{args.attack}_{args.eps:.4f}_n_it_{ls}_{test_cfg['NAME']}.pt" )
       
        with open(cfg['SAVE_DIR'] + "/test_results/main_results/"+ f"{strr}_numbers_{args.eps:.4f}_{test_cfg['NAME']}.txt", 'a+') as f:
            if ite == 0:
                f.write(f"{cfg['MODEL']['NAME']} - {test_cfg['BACKBONE']}\n")
                f.write(f"Clean results: {clean_stats}\n")
                f.write(f"{str(test_cfg['MODEL_PATH'])}\n")
            if args.adversarial:
                f.write(f"----- Linf radius: {args.eps:.4f} ------")
                f.write(f"Attack: {args.attack_type} {args.attack} \t \t Iterations: {args.n_iter} \t alpha: 0.01 \n")   
                f.write(f"Adversarial results: {adv_stats}\n") 
                # f.write(f"Adversarial mIoU: {adv_miou:.2%} \t mAcc: {adv_macc:.2%}\t aAcc: {adv_aacc:.2%}\n")
            f.write("\n")
        console.rule(f"[cyan]Segmentation results are saved in {cfg['SAVE_DIR']}" + "/test_results/"+ f"{strr}_numbers_{test_cfg['NAME']}.txt")

        # if args.store_data:
        #     save_dict = {'images': torch.cat(preds), 'labels': torch.cat(lblss)}
        #     torch.save(save_dict, cfg['SAVE_DIR'] + f"/test_results/adv_data_eps_{args.eps: .4f}_{str(cfg['MODEL']['BACKBONE'])}.pt")
        #     console.rule(f"[violet]Adversarial images and labels stored: {cfg['SAVE_DIR']}" + f"/test_results/adv_data_eps_{args.eps: .4f}_iter_{args.n_iter}_{str(cfg['MODEL']['BACKBONE'])}.pt")