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
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tools.val import Pgd_Attack, clean_accuracy
from PIL import Image, ImageDraw, ImageFont
import gc
from autoattack.other_utils import check_imgs

from functools import partial
from semseg.utils.visualize import generate_palette
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer, create_optimizers, adjust_learning_rate
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, Logger, makedir
from val import evaluate
import semseg.utils.attacker as attacker
import semseg.datasets.transform_util as transform
from semseg.metrics import Metrics

# from mmcv.utils import Config
# from mmcv.runner import get_dist_info
from semseg.datasets.dataset_wrappers import *
console = Console()
SEED = 225


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(SEED)
    random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)


def IoUAcc(y_trg, y_pred, classes = 21):

    trg = y_trg.squeeze(1)
    pred = y_pred
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    trg = trg.view(-1)
    for sem_class in range(classes): # loop over each class for IoU calculation
        pred_inds = (pred == sem_class)
        target_inds = (trg == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
            #print('Class {} IoU is {}'.format(class_names[sem_class+1], iou_now))
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            #print('Class {} IoU is {}'.format(class_names[sem_class+1], iou_now))
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
        
    
    return np.mean(present_iou_list)*100



def clean_accuracy(model, data_loder, n_batches=-1, n_cls=21):
    """Evaluate accuracy."""

    model.eval()
    acc = 0
    acc_cls = torch.zeros(n_cls)
    n_ex = 0
    n_pxl_cls = torch.zeros(n_cls)
    int_cls = torch.zeros(n_cls)
    union_cls = torch.zeros(n_cls)

    metrics = Metrics(n_cls, -1, 'cpu')

    for i, (input, target) in enumerate(data_loder):
        #print(input[0, 0, 0, :10])
        input = input.cuda()

        with torch.no_grad():
            output = model(input)

        #metrics.update(output.cpu(), target)

        pred = output.cpu().max(1)[1]
        acc_curr = pred == target

        # Compute correctly classified pixels for each class.
        for cl in range(n_cls):
            ind = target == cl
            acc_cls[cl] += acc_curr[ind].float().sum()
            n_pxl_cls[cl] += ind.float().sum()
        #print(acc_cls, n_pxl_cls)
        ind = n_pxl_cls > 0
        m_acc = (acc_cls[ind] / n_pxl_cls[ind]).mean()

        # Compute overall correctly classified pixels.
        acc_curr = acc_curr.float().view(input.shape[0], -1).mean(-1)
        acc += acc_curr.sum()
        n_ex += input.shape[0]

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
            f'batch={i} running mAcc={m_acc:.2%} batch aAcc={acc_curr.mean():.2%}',
            f' running mIoU={m_iou:.2%}')

        #print(metrics.compute_iou()[1], metrics.compute_pixel_acc()[1])

        if i + 1 == n_batches:
            break

    print(f'mAcc={m_acc:.2%} aAcc={acc / n_ex:.2%} mIoU={m_iou:.2%} ({n_ex} images)')
    return m_acc, acc / n_ex, m_iou
    #print(acc_cls / n_pxl_cls)
    #print(acc_cls.sum() / n_pxl_cls.sum())


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
        print(f'batch={i} avg. pixel acc={acc.mean():.2%}')

        adv_loader.append((x_adv.cpu(), target.cpu().clone()))
        if i + 1 == n_batches:
            break

    return adv_loader



def get_val_data(dataset_cfg, test_cfg):

    value_scale = 1.
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]


    
    if test_cfg['NAME'] == 'pascalvoc':
        val_transform = transform.Compose([
            transform.Crop([473, 473], crop_type='center', padding=mean, ignore_label=0),
            transform.ToTensor(),
            transform.Normalize(mean=[0, 0, 0], std=[255, 255, 255])  # To have images in [0, 1].
            ])
        data_l = '/data/naman_deep_singh/sem_seg/assests/pascalvoc_val.txt'

        val_dataset = get_segmentation_dataset(test_cfg['NAME'], split='val', data_root=dataset_cfg['ROOT'], data_list=data_l, transform=val_transform)
    else:
        input_transform = transforms.Compose([
        transforms.ToTensor()
        ])
        data_kwargs = {'transform': input_transform, 'base_size': 512, 'crop_size': [473, 473]}
        val_dataset = get_segmentation_dataset(test_cfg['NAME'], root=dataset_cfg['ROOT'], split='val', mode='val', **data_kwargs)

    return val_dataset

# alpha, num_iters
attack_setting = {'pgd': (0.01, 40), 'segpgd': (0.01, 40),
                    'cospgd': (0.15, 40),
                    'maskpgd': (0.15, 40)}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k_convnext_vena.yaml')
    parser.add_argument('--eps', type=float, default=8.)
    parser.add_argument('--store-data', action='store_true', help='PGD data?', default=False)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--adversarial', action='store_true', help='adversarial eval?', default=False)
    parser.add_argument('--attack', type=str, default='cospgd-loss', help='cospgd-loss, ce-avg or mask-ce-avg?')

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    dataset_cfg, model_cfg, test_cfg = cfg['DATASET'], cfg['MODEL'], cfg['EVAL']

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], dataset_cfg['N_CLS'],None)
    model.load_state_dict(torch.load(test_cfg['MODEL_PATH'], map_location='cpu'))

    model = model.to('cuda')

    val_data = get_val_data(dataset_cfg, test_cfg)
    dataloader = DataLoader(val_data, shuffle=True, batch_size=test_cfg['BATCH_SIZE'], worker_init_fn=seed_worker, generator=g)

    # clean_accuracy(model, dataloader)
    # exit()
    console.print(f"Model > [yellow1]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/yellow1]")
    console.print(f"Dataset > [yellow1]{test_cfg['NAME']}[/yellow1]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)

    preds = []
    lblss = []
    metrics = Metrics(dataset_cfg['N_CLS'], -1, 'cuda')

    macc, aacc, miou = clean_accuracy(model, dataloader, n_batches=20, n_cls=dataset_cfg['N_CLS'])
    
    if args.adversarial:
        strr = f'adversarial_{args.attack}'
    else:
        strr = 'clean'

    if args.adversarial:
        n_batches = 10
        # norm = 'Linf'
        args.norm = 'Linf'

        if args.norm == 'Linf' and args.eps >= 1.:
            args.eps /= 255.
        attack_fn = partial(
            attacker.apgd_restarts,
            norm=args.norm,
            eps=args.eps,
            n_iter=args.n_iter,
            n_restarts=1,
            use_rs=True,
            loss=args.attack if args.attack else 'ce-avg',
            verbose=True,
            track_loss='ce-avg',    
            log_path=None,
            )

        adv_loader = evaluate(dataloader, model, attack_fn, n_batches, args)
    

    if args.adversarial:
        adv_macc, adv_aacc, adv_miou = clean_accuracy(model, adv_loader, n_batches)

   
    with open(cfg['SAVE_DIR'] + "/test_results/"+ f"{strr}_numbers_{test_cfg['NAME']}.txt", 'a+') as f:

        f.write(f"{cfg['MODEL']['NAME']} - {cfg['MODEL']['BACKBONE']}\n")
        f.write(f"{str(test_cfg['MODEL_PATH'])}\n")
        f.write(f"Clean mIoU: {miou:.2%} \t mAcc: {macc:.2%}\t aAcc: {aacc:.2%}\n")

        if args.adversarial:
            f.write(f"Attack: APGD {args.attack} \t Linf radius: {args.eps:.4f} \t Iterations: {args.n_iter}\n")    
            f.write(f"Adversarial mIoU: {adv_miou:.2%} \t mAcc: {adv_macc:.2%}\t aAcc: {adv_aacc:.2%}\n")
        f.write("\n")
    console.rule(f"[cyan]Segmentation results are saved in {cfg['SAVE_DIR']}" + "/test_results/"+ f"{strr}_numbers_{test_cfg['NAME']}.txt")

    # if args.store_data:
    #     save_dict = {'images': torch.cat(preds), 'labels': torch.cat(lblss)}
    #     torch.save(save_dict, cfg['SAVE_DIR'] + f"/test_results/adv_data_eps_{args.eps: .4f}_{str(cfg['MODEL']['BACKBONE'])}.pt")
    #     console.rule(f"[violet]Adversarial images and labels stored: {cfg['SAVE_DIR']}" + f"/test_results/adv_data_eps_{args.eps: .4f}_iter_{args.n_iter}_{str(cfg['MODEL']['BACKBONE'])}.pt")