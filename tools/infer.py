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
from tools.val import pgd
from PIL import Image, ImageDraw, ImageFont
import gc

from semseg.utils.visualize import generate_palette
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer, create_optimizers, adjust_learning_rate
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, Logger, makedir
from val import evaluate
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




def get_val_data(dataset_cfg):

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': 512, 'crop_size': [473, 473]}


    val_dataset = get_segmentation_dataset(dataset_cfg['NAME'], root=dataset_cfg['ROOT'], split='val', mode='val', **data_kwargs)

    return val_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k_convnext_vena.yaml')
    parser.add_argument('--eps', type=float, default=4./255.)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--store-data', action='store_true', help='PGD data?', default=False)
    parser.add_argument('--adversarial', action='store_true', help='adversarial eval?', default=True)

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    dataset_cfg, model_cfg, test_cfg = cfg['DATASET'], cfg['MODEL'], cfg['EVAL']

    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], dataset_cfg['N_CLS'],None)
    model.load_state_dict(torch.load(test_cfg['MODEL_PATH'], map_location='cpu'))
    model = model.to('cuda')

    val_data = get_val_data(dataset_cfg)
    dataloader = DataLoader(val_data, shuffle=True, batch_size=test_cfg['BATCH_SIZE'], worker_init_fn=seed_worker, generator=g)

    console.print(f"Model > [yellow1]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/yellow1]")
    console.print(f"Dataset > [yellow1]{cfg['DATASET']['NAME']}[/yellow1]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)

    preds = []
    lblss = []
    metrics = Metrics(dataset_cfg['N_CLS'], -1, 'cuda')
    # metrics_clean = Metrics(dataset_cfg['N_CLS'], -1, 'cuda')

    if args.adversarial:
        strr = 'adversarial'
    else:
        strr = 'clean'

    for iterr, (img, lbl, _) in enumerate(dataloader):
        img = img.to('cuda')
        lbl = lbl.to('cuda')
        # print(lbl.min(), lbl.max())
        if args.adversarial:
            model.train()
            delta1 = pgd(model, img, lbl, epsilon=args.eps, alpha=1e2, num_iter=args.n_iter) # Various values of epsilon, alpha can be used to play with.
            model.eval()
            tensorr = (img.float() + delta1.float())
            print(delta1[0,0][lbl[0]==-1])
            ypa1 = model(tensorr, lbl)
            metrics.update(ypa1.softmax(dim=1), lbl)
        else:
            model.eval()
            ypa_c = model(img.float(), lbl)
            metrics.update(ypa_c.softmax(dim=1), lbl)

        if args.store_data:
            preds.append(tensorr.detach().cpu())
            lblss.append(lbl.detach().cpu())

        if args.adversarial and iterr == 20:
            break

    iuo_c, miou_c = metrics.compute_iou()
    cla_acc, macc, aacc = metrics.compute_pixel_acc()

    with open(cfg['SAVE_DIR'] + "/test_results/"+ f"{strr}_numbers_{dataset_cfg['NAME']}.txt", 'a+') as f:
        f.write(f"{cfg['MODEL']['NAME']} - {cfg['MODEL']['BACKBONE']}\n")
        f.write(f"Clean mIoU {miou_c} \n class IoU {iuo_c}\n")
        f.write(f"Class acc {cla_acc} \n mAcc {macc}, aAcc {aacc}\t")
        f.write("\n")
    console.rule(f"[cyan]Segmentation results are saved in {cfg['SAVE_DIR']}" + "/test_results/"+ f"{strr}_numbers_{dataset_cfg['NAME']}.txt")

    if args.store_data:
        save_dict = {'images': torch.cat(preds), 'labels': torch.cat(lblss)}
        torch.save(save_dict, cfg['SAVE_DIR'] + f"/test_results/adv_data_eps_{args.eps: .4f}_{str(cfg['MODEL']['BACKBONE'])}.pt")
        console.rule(f"[violet]Adversarial images and labels stored: {cfg['SAVE_DIR']}" + f"/test_results/adv_data_eps_{args.eps: .4f}_iter_{args.n_iter}_{str(cfg['MODEL']['BACKBONE'])}.pt")