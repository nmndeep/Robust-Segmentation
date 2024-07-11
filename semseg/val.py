import math

import torch
import torch.nn as nn
from tabulate import tabulate
from torch.nn import functional as F
from tqdm import tqdm

from semseg.datasets import *
from semseg.metrics import Metrics
from semseg.models import *


@torch.no_grad()
def evaluate(model, dataloader, device, cls, n_batches=-1):
    print("Evaluating...")
    # model.freeze_bn()
    model.eval()
    metrics = Metrics(cls, -1, device)

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images)
        metrics.update(preds.softmax(dim=1), labels)
        if i + 1 == n_batches:
            break
    ious, miou = metrics.compute_iou()
    cla_acc, macc, aacc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    return cla_acc, macc, aacc, f1, mf1, ious, miou


def js_div_fn(p, q, softmax_output=False, reduction="none", red_dim=None):
    """
    Compute JS divergence between p and q.

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

    loss = (
        F.kl_div(m.log(), p, reduction=reduction)
        + F.kl_div(m.log(), q, reduction=reduction)
    ) / 2
    if red_dim is not None:
        assert reduction == "none", "Incompatible setup."
        loss = loss.sum(dim=red_dim)

    return loss


def attack_pgd_training(
    model,
    X,
    y,
    eps,
    alpha,
    opt,
    half_prec,
    attack_iters,
    rs=True,
    early_stopping=False,
):
    delta = torch.zeros_like(X).cuda()
    if rs:
        delta.uniform_(-eps, eps)

    delta.requires_grad = True
    for _ in range(attack_iters):
        output = model((X + delta, 0, 1))
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


def js_loss(p, q, reduction="mean"):
    loss = js_div_fn(p, q, red_dim=(1))  # Sum over classes.
    if reduction == "mean":
        return loss.view(p.shape[0], -1).mean(-1)
    elif reduction == "none":
        return loss


def masked_cross_entropy(pred, target):
    """Cross-entropy of only correctly classified pixels."""
    mask = pred.max(1)[1] == target
    loss = F.cross_entropy(pred, target, reduction="none")
    loss = (mask.float() * loss).view(pred.shape[0], -1).mean(-1)

    return loss


losses = {
    "pgd": lambda x, y: F.cross_entropy(x, y),
    "mask-ce-avg": masked_cross_entropy,
    "js-avg": js_loss,
    "l2-loss": lambda x, y: ((x - y) ** 2).view(x.shape[0], -1).sum(-1),
    # 'l2-loss': lambda x, y: F.mse_loss(x, y)
}


class Pgd_Attack:
    def __init__(self, eps=4.0 / 255.0, alpha=1e-2, num_iter=2, los="pgd"):
        self.epsilon = eps
        self.num_iter = num_iter
        self.loss_fn = losses[los]
        self.los_name = los
        self.alpha = alpha

    def adv_attack(self, model, X, y, wt=None):  # Untargetted Attack
        model.eval()
        # x_best_adv = x_adv.clone()
        delta = torch.zeros_like(X).cuda()
        delta.requires_grad = True
        running_best_loss = torch.zeros(X.size(0)).cuda()
        # trg = y.squeeze(1)
        best_delta = torch.zeros_like(X)
        for t in range(self.num_iter):
            lam_t = t / 2 * self.num_iter
            # TODO fix for consistency
            # logits = model(input=(X + delta).clamp(0., 1.))
            logits = model((X + delta).clamp(0.0, 1.0))

            # print(t)
            if self.los_name == "segpgd-loss":
                loss = self.loss_fn(logits, y.long(), t, self.num_iter)
            else:
                loss = self.loss_fn(logits, y.long())

            ind_pred = (loss.detach().clone() >= running_best_loss).nonzero().squeeze()
            # print(ind_pred)
            # print(loss[ind_pred[2]])
            running_best_loss[ind_pred] = (
                loss[ind_pred].detach().clone() + 0.0
            )  # .detach().clone()
            # print(running_best_loss[ind_pred].size())
            # print(delta[ind_pred].data.size())
            loss = loss.sum()
            loss.backward()
            grad = delta.grad.detach()
            grad_sign = torch.sign(grad)
            delta.data = delta + self.alpha * grad_sign
            delta.data = (X + delta.data).clamp(0.0, 1.0) - X
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()
            delta.detach()
            best_delta[ind_pred] = delta[ind_pred].data  # .detach().clone()

        x_adv = (X + best_delta).clamp(0.0, 1.0)
        return x_adv.detach(), None, None


class Pgd_Attack_1:
    def __init__(self, epsilon=4.0 / 255.0, alpha=1e-2, num_iter=2, los="pgd"):
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.loss_fn = losses[los]
        self.los_name = los
        self.alpha = alpha

    def adv_attack(self, model, X, y):  # Untargetted Attack
        model.eval()
        # x_best_adv = x_adv.clone()
        delta = torch.zeros_like(X).cuda()
        if True:  # random restart
            delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True
        # trg = y.squeeze(1)
        for t in range(self.num_iter):
            lam_t = t / 2 * self.num_iter
            # TODO fix for consistency
            logits = model(X + delta)
            # logits = model((X + delta).clamp(0., 1.))            # print(t)
            if self.los_name == "segpgd-loss":
                loss = self.loss_fn(logits, y, t, self.num_iter)
            else:
                loss = self.loss_fn(logits, y)

            loss = loss.sum()
            loss.backward()
            grad = delta.grad.detach()
            grad_sign = torch.sign(grad)
            delta.data = delta + self.alpha * grad_sign
            delta.data = (X + delta.data).clamp(0.0, 1.0) - X
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon)
            delta.grad.zero_()
            delta.detach()

        x_adv = (X + delta).clamp(0.0, 1.0)
        return x_adv.detach(), logits, None


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
        # print(acc_cls, n_pxl_cls)
        ind = n_pxl_cls > 0
        m_acc = (acc_cls[ind] / n_pxl_cls[ind]).mean()

        # Compute overall correctly classified pixels.
        acc_curr = acc_curr.float().view(input.shape[0], -1).mean(-1)
        acc += acc_curr.sum()
        n_ex += input.shape[0]
        print(acc_curr.shape)

        print(f"batch={i} running mAcc={m_acc:.2%} batch aAcc={acc_curr.mean():.2%}")

        if i + 1 == n_batches:
            break

    print(f"mAcc={m_acc:.2%} aAcc={acc / n_ex:.2%} ({n_ex} images)")


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer(
            "negatives_mask",
            (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float(),
        )

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # print(emb_i.size())
        emb_i_fl = emb_i.flatten(start_dim=1)
        emb_j_fl = emb_j.flatten(start_dim=1)
        z_i = F.normalize(emb_i_fl, dim=1)
        z_j = F.normalize(emb_j_fl, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        print(representations.size())
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature
        )

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


def ce_unsup(out, targets, reduction="mean", targeted=False, alpha=0.0):
    out = out.flatten(start_dim=1)
    targets = targets.flatten(start_dim=1)
    assert out.shape[0] == targets.shape[0]
    assert out.shape[0] > 1
    assert out.shape[1] == targets.shape[1]
    # assert out.shape[1] in (512, 768)
    assert targeted if alpha > 0.0 else True, (targeted, alpha)

    preds = out @ targets.T
    labels = torch.arange(out.shape[0]).to(out.device)
    if targeted:
        if alpha == 0.0:
            labels = (labels + 1) % out.shape[0]
        elif alpha > 0.0:
            assert alpha == 1.0
            # find which embeddings are already very close
            # by finding the maximal off-diagonal value in preds
            labels = torch.argmax(preds - 10 * torch.diag(preds.diag()), dim=1)
        else:
            raise ValueError(f"alpha={alpha} not supported")
    if reduction == "mean":
        loss = F.cross_entropy(preds, labels)
    elif reduction == "none":
        loss = F.cross_entropy(preds, labels, reduction="none")
    loss = -loss if targeted else loss
    return loss


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
            new_H, new_W = (
                int(math.ceil(new_H / 32)) * 32,
                int(math.ceil(new_W / 32)) * 32,
            )
            scaled_images = F.interpolate(
                images,
                size=(new_H, new_W),
                mode="bilinear",
                align_corners=True,
            )
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            logits = F.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=True
            )
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(
                    logits, size=(H, W), mode="bilinear", align_corners=True
                )
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)

    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou

    print(tabulate(table, headers="keys"))
