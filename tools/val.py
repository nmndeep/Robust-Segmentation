import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

@torch.no_grad()
def evaluate(model, dataloader, device, cls):
    print('Evaluating...')
    # model.freeze_bn()
    model.eval()
    metrics = Metrics(cls, -1, device)

    for i, (images, labels, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        preds = model(pixel_values=images, labels=labels)
        metrics.update(preds.softmax(dim=1), labels)
    
    ious, miou = metrics.compute_iou()
    cla_acc, macc, aacc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return acc, macc, aacc, f1, mf1, ious, miou


def pgd(model, X, y, epsilon, alpha, num_iter): # Untargetted Attack
    
    delta = torch.zeros_like(X, requires_grad=True)
    trg = y.squeeze(1)
    
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss(ignore_index = -1)(model(X + delta, y)[1], trg.long())
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    print('Loss after iteration {}: {:.2f}'.format(t+1, loss.item()))
    return delta.detach()



@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform)
    dataloader = DataLoader(dataset, 1, num_workers=1, pin_memory=True)

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes)
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model = model.to(device)

    if eval_cfg['MSF']['ENABLE']:
        acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
    else:
        acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device)

    table = {
        'Class': list(dataset.CLASSES) + ['Mean'],
        'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Acc': acc + [macc]
    }

    print(tabulate(table, headers='keys'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/custom.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)