
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
from functools import partial


@torch.no_grad()
def evaluate(model, dataloader, device, cls, n_batches=-1):
    print('Evaluating...')
    # model.freeze_bn()
    model.eval()
    metrics = Metrics(cls, -1, device)

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        preds = model(input=images)
        metrics.update(preds.softmax(dim=1), labels)
        if i + 1 == n_batches:
            break
    ious, miou = metrics.compute_iou()
    cla_acc, macc, aacc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    
    return cla_acc, macc, aacc, f1, mf1, ious, miou

def segpgd_loss(pred, target, lam):

    # print(pred.shape)
    corr_mask = pred.max(1)[1] == target
    wron_mask = pred.max(1)[1] == target
    pred_corr = pred[corr_mask]
    pred_wrong = pred[wron_mask]
    sh = target.shape
    l_j = F.cross_entropy(pred_corr, target[corr_mask], reduction='none').view(sh[0], -1).mean(-1)
    l_k = F.cross_entropy(pred_wrong, target[wron_mask], reduction='none').view(sh[0], -1).mean(-1)
    loss = (((1-lam) * l_j  + lam * l_k)/(sh[1]*sh[2])) #.view(sh[0], -1).mean(-1)
    return loss

def js_div_fn(p, q, softmax_output=False, reduction='none', red_dim=None):
    """Compute JS divergence between p and q.

    p: logits [bs, n_cls, ...]
    q: labels [bs]
    softmax_output: if softmax has already been applied to p
    reduction: to pass to KL computation
    red_dim: dimensions over which taking the sum
    """
    
    if not softmax_output:
        p = F.softmax(p, 1)
    q = F.one_hot(q.view(q.shape[0], -1), p.shape[1])
    q = q.permute(0, 2, 1).view(p.shape).float()
    
    m = (p + q) / 2
    
    loss = (F.kl_div(m.log(), p, reduction=reduction)
            + F.kl_div(m.log(), q, reduction=reduction)) / 2
    if red_dim is not None:
        assert reduction == 'none', 'Incompatible setup.'
        loss = loss.sum(dim=red_dim)
    
    return loss

def cospgd_loss(pred, target, reduction='mean'):
    """Implementation of the loss for semantic segmentation from
    https://arxiv.org/abs/2302.02213.

    pred: B x cls x h x w
    target: B x h x w
    """

    #with torch.no_grad():
    sigm_pred = torch.sigmoid(pred)
    sh = target.shape
    n_cls = pred.shape[1]
    y = F.one_hot(target.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    #w = (sigm_pred * y).sum(1) / pred.norm(p=2, dim=1) #sigm_pred.max(dim=1)[0] #pred.norm(p=2, dim=1)
    w = F.cosine_similarity(sigm_pred, y)
    loss = F.cross_entropy(pred, target, reduction='none')
    #with torch.no_grad():
    loss = w.detach() * loss

    if reduction == 'mean':
        return loss.view(sh[0], -1).mean(-1)

    return loss


def segpgd_loss(pred, target, t, max_t, reduction='mean'):
    """Implementation of the loss of https://arxiv.org/abs/2207.12391.

    pred: B x cls x h x w
    target: B x h x w
    t: current iteration
    max_t: total iterations
    """

    lmbd = t / 2 / max_t
    corrcl = (pred.max(1)[1] == target).float()
    loss = F.cross_entropy(pred, target, reduction='none')
    loss = (1 - lmbd) * corrcl * loss + lmbd * (1 - corrcl) * loss

    if reduction == 'mean':
        return loss.view(target.shape[0], - 1).mean(-1)
    return loss


def attack_pgd_training(model, X, y, eps, alpha, opt, half_prec, attack_iters, rs=True, early_stopping=False):
    delta = torch.zeros_like(X).cuda()
    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model(     (X + delta, 0, 1))
        loss = F.cross_entropy(output, y)
        if half_prec:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
                delta.grad.mul_(loss.item() / scaled_loss.item())
        else:
            loss.backward()
        grad = delta.grad.detach()

        if early_stopping:
            idx_update = output.max(1)[1] == y
        else:
            idx_update = torch.ones(y.shape, dtype=torch.bool)
        grad_sign = sign(grad)
        delta.data[idx_update] = (delta + alpha * grad_sign)[idx_update]
        delta.data = clamp(X + delta.data, 0, 1) - X
        delta.data = clamp(delta.data, -eps, eps)
        delta.grad.zero_()

    return delta.detach()

def js_loss(p, q, reduction='mean'):

    loss = js_div_fn(p, q, red_dim=(1)) # Sum over classes.
    if reduction == 'mean':
        return loss.view(p.shape[0], -1).mean(-1)
    elif reduction == 'none':
        return loss

def masked_cross_entropy(pred, target):
    """Cross-entropy of only correctly classified pixels."""

    mask = pred.max(1)[1] == target
    loss = F.cross_entropy(pred, target, reduction='none')
    loss = (mask.float() * loss).view(pred.shape[0], -1).mean(-1)

    return loss

def single_logits_loss(pred, target, normalized=False, reduction='none',
        masked=False):
    """The (normalized) logit of the correct class is minimized."""

    if normalized:
        pred = pred / (pred ** 2).sum(1, keepdim=True).sqrt() #.detach()
    sh = target.shape
    n_cls = pred.shape[1]
    y = F.one_hot(target.view(sh[0], -1), n_cls)
    y = y.permute(0, 2, 1).view(pred.shape)
    loss = -1 * (y * pred).sum(1)
    if masked:
        mask = pred.max(1)[1] == target
        loss = mask.float().detach() * loss

    if reduction == 'mean':
        return loss.view(sh[0], -1).mean(-1)
    return loss

losses = {'pgd': lambda x, y: F.cross_entropy(x, y),
    'cospgd-loss': cospgd_loss, 
    'segpgd-loss': segpgd_loss,
    'norm-corrlog-avg': partial(single_logits_loss, normalized=True, reduction='mean'),
    'mask-ce-avg': masked_cross_entropy,
    'js-avg': js_loss
}


class Pgd_Attack():
    
    def __init__(self, epsilon=4./255., alpha=1e-2, num_iter=2, los='pgd'):
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.loss_fn = losses[los]
        self.los_name = los
        self.alpha = alpha

    def adv_attack(self, model, X, y): # Untargetted Attack
        
        model.eval()

        delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True
        # trg = y.squeeze(1)
        print(y.min(), y.max())
        for t in range(self.num_iter):
            lam_t = t / 2 * self.num_iter
            logits = model(input=(X + delta).clamp(0., 1.))
            # print(t)
            if self.los_name == 'segpgd-loss':
                loss = self.loss_fn(logits, y.long(), t, self.num_iter)
            else:
                loss = self.loss_fn(logits, y.long())
            loss = loss.sum()
            loss.backward()
            grad = delta.grad.detach()
            grad_sign = torch.sign(grad)
            delta.data = (delta + self.alpha * grad_sign)
            delta.data = (X + delta.data).clamp(0., 1.) - X
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()
            delta.detach()

        x_adv = (X + delta).clamp(0., 1.)
        return x_adv.detach(), None, None


def clean_accuracy(model, data_loder, n_batches=-1, n_cls=21):
    """Evaluate accuracy."""

    model.eval()
    acc = 0
    acc_cls = torch.zeros(n_cls)
    n_ex = 0
    n_pxl_cls = torch.zeros(n_cls)

    for i, (input, target, _) in enumerate(data_loder):
        input = input.cuda()

        with torch.no_grad():
            output = model(input)
        acc_curr = output.cpu().max(1)[1] == target
        # print(acc_curr.shape)
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
        print(acc_curr.shape)


        print(f'batch={i} running mAcc={m_acc:.2%} batch aAcc={acc_curr.mean():.2%}')

        if i + 1 == n_batches:
            break

    print(f'mAcc={m_acc:.2%} aAcc={acc / n_ex:.2%} ({n_ex} images)')




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
