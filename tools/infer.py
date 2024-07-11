import argparse
import random
import sys
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from rich.console import Console
from torch import Tensor
from tqdm import tqdm

import semseg.attacker as attacker
from semseg.datasets import *
from semseg.datasets.dataset_wrappers import *
from semseg.models import *
from semseg.utils.utils import *

from .worse_only import evalSEA

console = Console()
SEED = 225
random.seed(SEED)
np.random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)


def seed_worker(worker_id):
    worker_seed = SEED % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def check_imgs(adv, x, norm, verbose=False):
    #from AutoAttack
    delta = (adv - x).view(adv.shape[0], -1)
    if norm == 'Linf':
        res = delta.abs().max(dim=1)[0]
    elif norm == 'L2':
        res = (delta ** 2).sum(dim=1).sqrt()
    elif norm == 'L1':
        res = delta.abs().sum(dim=1)

    str_det = f'max {norm} pert: {res.max():.5f}, nan in imgs: {(adv != adv).sum()}, max in imgs: {adv.max():.5f}, min in imgs: {adv.min():.5f}'
    if verbose:
        print(str_det)
    
    return str_det


def eval_performance(
    model,
    data_loader,
    n_batches=-1,
    n_cls=21,
    return_output=False,
    ignore_index=-1,
    return_preds=False,
    verbose=False,
):
    """Evaluate accuracy and mIoU"""
    model.eval()
    acc_cls = torch.zeros(n_cls)
    n_ex = 0
    n_pxl_cls = torch.zeros(n_cls)
    int_cls = torch.zeros(n_cls)
    union_cls = torch.zeros(n_cls)

    l_output = []

    for i, vals in enumerate(data_loader):
        if False:
            print(i)
        else:
            input, target = vals[0], vals[1]

            input = input.to("cuda")

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

            # Compute intersection and union.
            intersection_all = pred == target
            for cl in range(n_cls):
                ind = target == cl
                int_cls[cl] += intersection_all[ind].float().sum()
                union_cls[cl] += (
                    ind.float().sum()
                    + (pred == cl).float().sum()
                    - intersection_all[ind].float().sum()
                )
            ind = union_cls > 0
            m_iou = (int_cls[ind] / union_cls[ind]).mean()

            if verbose:
                print(
                    f"batch={i} running mAcc={m_acc:.2%} running aAcc={a_acc.mean():.2%}",
                    f" running mIoU={m_iou:.2%}",
                )

        if i + 1 == n_batches:
            print("enough batches seen")
            break

    l_output = torch.cat(l_output)
    stats = {"mAcc": m_acc.item(), "aAcc": a_acc.item(), "mIoU": m_iou.item()}

    return stats, l_output


def evaluate(val_loader, model, attack_fn, n_batches=-1, args=None, weights=None):
    """Run attack on points."""
    model.eval()
    adv_loader = []

    for i, (input, target, _) in tqdm(enumerate(val_loader), desc='Attack'):
        # print(input[0, 0, 0, :10])
        input = input.to("cuda")
        target = target.to("cuda")

        x_adv, _, acc = attack_fn(model, input.clone(), target, weights)
        check_imgs(input, x_adv, norm=args.norm)
        if False:
            print(f"batch={i} avg. pixel acc={acc.mean():.2%}")

        adv_loader.append((x_adv.cpu(), target.cpu().clone()))
        if i + 1 == n_batches:
            break

    return adv_loader


def get_data(dataset_cfg, test_cfg):
    if str(test_cfg["NAME"]) == "pascalvoc":
        val_data = get_segmentation_dataset(
            test_cfg["NAME"],
            root=dataset_cfg["ROOT"],
            split="val",
            transform=torchvision.transforms.ToTensor(),
            base_size=512,
            crop_size=(473, 473),
        )

    elif str(test_cfg["NAME"]) == "pascalaug":
        val_data = get_segmentation_dataset(
            test_cfg["NAME"],
            root=dataset_cfg["ROOT"],
            split="val",
            transform=torchvision.transforms.ToTensor(),
            base_size=512,
            crop_size=(473, 473),
        )

    elif str(test_cfg["NAME"]).lower() == "ade20k":
        val_data = get_segmentation_dataset(
            test_cfg["NAME"],
            root=dataset_cfg["ROOT"],
            split="val",
            transform=torchvision.transforms.ToTensor(),
            base_size=520,
            crop_size=(512, 512),
        )

    else:
        raise ValueError("Unknown dataset.")

    return val_data


class MaskClass(nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input: Tensor) -> Tensor:
        if self.ignore_index == 0:
            return input[:, 1:]
        else:
            return torch.cat(
                (
                    input[:, : self.ignore_index],
                    input[:, self.ignore_index + 1 :],
                ),
                dim=1,
            )


def mask_logits(model: nn.Module, ignore_index: int) -> nn.Module:
    # TODO: adapt for list of indices.
    layers = OrderedDict([("model", model), ("mask", MaskClass(ignore_index))])
    return nn.Sequential(layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/pascalvoc_convnext.yaml")
    parser.add_argument("--eps", type=float, default=8.0)
    parser.add_argument(
        "--n_iter", type=int, default=300
    )  # donot change set to the standard SEA implementaiton
    parser.add_argument(
        "--adversarial",
        action="store_true",
        help="adversarial eval?",
        default=True,
    )
    parser.add_argument(
        "--attack",
        type=str,
        default=None,
        help="mask-ce-avg, js-avg, mask-ce-bal?",
    )  # overwritten later
    parser.add_argument(
        "--n_batches", type=int, default=-1
    )  # set to -1 for full validation-sets
    parser.add_argument(
        "--cleanup", type=int, default=1
    )  # remove intermediate I/O saved files 

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    dataset_cfg, model_cfg, test_cfg = (
        cfg["DATASET"],
        cfg["MODEL"],
        cfg["EVAL"],
    )

    if model_cfg["NAME"] == "SegMenter":
        model_cfg1, _ = load_config_segmenter(
            backbone=model_cfg["BACKBONE"], n_cls=test_cfg["N_CLS"]
        )
        model = create_segmenter(
            model_cfg1, model_cfg["PRETRAINED"], test_cfg["BACKBONE"]
        )

    elif model_cfg["NAME"] == "UperNetForSemanticSegmentation":
        model = eval(model_cfg["NAME"])(test_cfg["BACKBONE"], test_cfg["N_CLS"], None)

    else:
        model = eval(model_cfg["NAME"])(50, test_cfg["N_CLS"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(test_cfg["MODEL_PATH"], map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device)

    los_pairs = [
        "mask-ce-bal",
        "mask-ce-avg",
        "js-avg",
    ]  # standard-SEA attacks

    model.eval()

    val_data = get_data(dataset_cfg, test_cfg)
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=test_cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=None,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # class balance pre-computed weights
    bal_weights = (
        torch.tensor(ADE_WTS)
        if test_cfg["NAME"].lower() == "ade20k"
        else torch.tensor(VOC_WTS)
    )

    make_attack_dirs(cfg["SAVE_DIR"])  # noqa: F405
    modelName = getModelName(model_cfg["NAME"], test_cfg["BACKBONE"])

    console.print(f"Model > [yellow1]{modelName}[/yellow1]")
    console.print(f"Dataset > [yellow1]{test_cfg['NAME']}[/yellow1]")

    save_argmax_of_log = True  # set to true for saving argmax of image-wise logits after adversarial attack
    verbose = False  # print all intermediate step computations?
    n_batches = args.n_batches

    # check the clean performance of the model
    clean_stats, _ = eval_performance(
        model,
        val_data_loader,
        n_batches=n_batches,
        n_cls=test_cfg["N_CLS"],
        ignore_index=-1,
        verbose=verbose,
    )
    console.print(f"Clean performance. > [cyan]{clean_stats}[/cyan]")

    if not args.adversarial:
        sys.exit()

    args.norm = "Linf"
    test_eps = args.eps
    loss_wise_logits = []
    indiv_mious = []

    for ite, loss_ in enumerate(los_pairs):
        console.print(
            f"Loss > [yellow1]{loss_}[/yellow1]"
            + f" Epsilon > [yellow1]{test_eps}[/yellow1]"
        )
        args.attack = loss_
        attack_fn = partial(
            attacker.apgd_largereps,
            norm=args.norm,
            eps=test_eps / 255.0,
            n_iter=args.n_iter,
            n_restarts=1,  # args.n_restarts,
            use_rs=True,
            loss=args.attack if args.attack else "ce-avg",
            verbose=verbose,
            track_loss="ce-avg",
            log_path=None,
            num_classes=test_cfg["N_CLS"],
            early_stop=True,
        )
        adv_loader = evaluate(
            val_data_loader, model, attack_fn, n_batches, args, bal_weights
        )

        adv_stats, l_outs = eval_performance(
            model,
            adv_loader,
            n_batches,
            n_cls=dataset_cfg["N_CLS"],
            ignore_index=-1,
            verbose=verbose,
        )

        # Save argmax for later computaiton of worse case aAcc and mIoUs
        if save_argmax_of_log:
            torch.save(
                l_outs,
                cfg["SAVE_DIR"] + f"/argmax-logs/{modelName}_{loss_}_{test_eps}.pt",
            )

        loss_wise_logits.append(l_outs.detach().cpu())

        indiv_mious.append(adv_stats["mIoU"])

        writeIndivloss(
            cfg["SAVE_DIR"], modelName, clean_stats, test_eps, loss_, adv_stats
        )
        print(f"{loss_} evaluation completed")

    addendum = "SEA_" + modelName
    save_dict = {
        "seed": SEED,
        "worst_Acc": 0,
        "worst_Acc_indiv": 0,
        # "final_matrix_Acc": 0,
        "final_miou": 0,
        "loss-wise_miou": indiv_mious,
    }
    console.print(f"Computing worse-cases > [red]{addendum}[/red]")

    evall = evalSEA(
        val_data=val_data,
        l_outs=loss_wise_logits,
        eps=test_eps,
        n_cls=test_cfg["N_CLS"],
        addendum=addendum,
        saveDir=cfg["SAVE_DIR"],
        saveDict=save_dict,
        modelName=modelName,
    )
    evall.worse_case_eval(bs=test_cfg["BATCH_SIZE"], n_batches=n_batches)
    evall.worst_case_miou()
    # Save final numbers:
    torch.save(
        evall.saveDict,
        cfg["SAVE_DIR"] + f"/worse_{addendum}_{test_cfg['NAME']}_{test_eps}.pt",
    )
    
    console.print(f"Attack-wsie statistics at: > [red]{cfg["SAVE_DIR"] + "sea-stats"}[/red]")
    console.print(f"SEA-statistics at: > [red]{cfg["SAVE_DIR"] + f"worse_{addendum}_{test_cfg['NAME']}_{test_eps}.pt"}[/red]")

    if bool(args.cleanup):
        remove_dirs(cfg["SAVE_DIR"])