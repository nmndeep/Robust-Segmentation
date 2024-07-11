import os
import random
import statistics

import numpy as np
import torch
import torch.utils.data as data
from rich.console import Console
from tqdm import tqdm

from semseg.datasets.dataset_wrappers import *  # noqa: F403

console = Console()
SEED = 225
random.seed(SEED)
np.random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)


def seed_worker(worker_id):
    # Seed everything
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def update_fn(data, target, running_stats, stat="intersection", n_cls=21):
    intersection_all1 = data.squeeze() == target.squeeze()

    if stat == "intersection":
        for cl in range(n_cls):
            ind = (target == cl).squeeze()
            running_stats[cl] += intersection_all1[ind].float().sum()
    else:
        for cl in range(n_cls):
            ind = (target == cl).squeeze()
            running_stats[cl] += (
                ind.float().sum()
                + (data.squeeze() == cl).float().sum()
                - intersection_all1[ind].float().sum()
            )

    return running_stats


def update_fn_indiv(data, target, stat="intersection", n_cls=21):
    intersection_all1 = data.squeeze() == target.squeeze()
    indiv_stats = torch.zeros(n_cls)

    if stat == "intersection":
        for cl in range(n_cls):
            ind = (target == cl).squeeze()
            indiv_stats[cl] = intersection_all1[ind].float().sum()
    else:
        for cl in range(n_cls):
            ind = (target == cl).squeeze()
            indiv_stats[cl] = (
                ind.float().sum()
                + (data.squeeze() == cl).float().sum()
                - intersection_all1[ind].float().sum()
            )

    return indiv_stats


def _compute_miou(inters, union):
    iou = []
    for a, b in zip(inters, union):
        if b == 0:  # Skip empty classes.
            continue
        iou.append(a.item() / b.item())

    return statistics.mean(iou)


def _compute_miou_subtraction(running_int, running_union, inters, union):
    iou = []
    uni = []
    miou = []
    ct = 0
    for a, b, c, d in zip(running_int, running_union, inters, union):
        # print(a,b,c,d)
        if b == 0:  # Skip empty classes.
            continue
        iou.append(a.item() + c.item())
        uni.append(b.item() + d.item())
        miou.append(iou[ct] / (uni[ct] + 1e-8))
        ct += 1

    return statistics.mean(miou), iou, uni


class evalSEA:
    """
    A class to do worse-case SEA evalution

    ...

    Attributes
    ----------
    val_data : Object
        a dataset object for the dataset to be evaluated on
    l_outs : list
        loss wise logits per image
    eps : float
        perturbation strength
    n_cls : int
        number of classes in the dataset
    addendum: str
        to locate the necessary  files on disk
    saveDir: str
        output directory location, where statistics will be saved and read from
    saveDict: dict
        output statistics will be saved in this dict, initialized with loss-wise mIoUs
    modelName: str
        modelname to locate necessary I/O files

    Methods
    -------
    get_loader(bs)
        create dataloader for val_data object
        Parameters
        ----------
        bs : int, batch-size, set to 1 for worse-mIoU
    
    worst_case_miou()
        Computes point-wise worce mIoU

    worse_case_eval(bs)
        
        Computes worse aACC across attacks
        Parameters
        ----------

        bs : int, batch-size, 
        n_batches : int, number of batches to evaluate for : set to -1 for all

    """ 

    def __init__(
        self,
        val_data,
        l_outs,
        eps,
        n_cls,
        addendum,
        saveDir,
        saveDict,
        modelName,
    ):
        self.val_data = val_data
        self.l_output = l_outs
        self.eps = eps
        self.addendum = addendum
        self.saveDir = saveDir
        self.saveDict = saveDict
        self.modelName = modelName
        self.n_cls = n_cls
        self.los_pairs = [
            "mask-ce-bal",
            "mask-ce-avg",
            "js-avg",
        ]  # standard-SEA attacks

    def get_loader(self, bs=1):
        val_loader = data.DataLoader(
            self.val_data,
            batch_size=bs,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            sampler=None,
            worker_init_fn=seed_worker,
            generator=g,
        )
        return val_loader

    def worst_case_miou(self):
        running_intersection = [0 for _ in range(self.n_cls)]
        running_union = [0 for _ in range(self.n_cls)]
        selected_attk_indx = []
        data_loader = self.get_loader(bs=1)  # bs = 1 since we do it image-wise

        if not self.l_output:
            self.l_output = []
            for loss_ in self.los_pairs:
                self.l_output.append(
                    torch.load(
                        self.saveDir + f"/argmax-logs/{self.modelName}_{loss_}_{self.eps}.pt"
                    )
                )

        # if running stats are already saved, load from directory
        if not os.path.isfile(
            self.saveDir + f"/running_stats/stats_{self.addendum}_{self.eps}.pt"
        ):
            class_wise_logits = torch.stack(
                self.l_output, dim=0
            )  # (number_losses, number-images, im_size)
            cons_ints = torch.zeros((
                class_wise_logits.shape[0],
                class_wise_logits.shape[1],
                self.n_cls,
            ))
            cons_unions = torch.zeros((
                class_wise_logits.shape[0],
                class_wise_logits.shape[1],
                self.n_cls,
            ))
            total_points = class_wise_logits.size(1)
            console.print("[light_steel_blue1] Starting worse-mIoU computation, point-wise [/light_steel_blue1]")

            for i, vals in tqdm(enumerate(data_loader), desc="Worse-mIoU"):
                _, target = vals[0], vals[1]

                pred = class_wise_logits[:, i, :, :]

                attack_to_select = None

                # iterate over each attack
                for attack in range(class_wise_logits.size(0)):
                    # Update statistics when using `attack` for the current image.
                    cons_ints[attack, i, :] = update_fn_indiv(
                        pred[attack].clone(),
                        target,
                        "intersection",
                        self.n_cls,
                    )
                    cons_unions[attack, i, :] = update_fn_indiv(
                        pred[attack].clone(), target, "union", self.n_cls
                    )

                    # always start with Mask-ce-Bal
                    attack_to_select = 0
                selected_attk_indx.append(attack_to_select)

                # Update running statistics with best attack.
                running_intersection = update_fn(
                    pred[0, :, :],
                    target,
                    running_intersection,
                    "intersection",
                    self.n_cls,
                )
                running_union = update_fn(
                    pred[0, ::], target, running_union, "union", self.n_cls
                )
                if i+1 >= total_points:
                    break

            savedict = {
                "run_int_imwise": cons_ints,
                "run_union_imwise": cons_unions,
                "run_intersect_abs": running_intersection,
                "run_union_abs": running_union,
            }
            del class_wise_logits

            torch.save(
                savedict,
                self.saveDir + f"/test_results/stats_{self.addendum}_{self.eps}.pt",
            )

        else:
            tenss = torch.load(
                self.saveDir + f"/test_results/stats_{self.addendum}_{self.eps}.pt"
            )
            cons_ints, cons_unions, running_intersection, running_union = (
                tenss["run_int_imwise"],
                tenss["run_union_imwise"],
                tenss["run_intersect_abs"],
                tenss["run_union_abs"],
            )

        # random starts
        final_miou = _compute_miou(running_intersection, running_union)
        console.print(
            f"Miou after mask-ce-bal > [light_steel_blue1]{final_miou}[/light_steel_blue1]"
        )
        n_rounds = 1000
        selected_attk_indx = [0] * cons_ints.shape[1]  # we start from mask-ce-bal
        console.print(
            f"[cyan]Starting {n_rounds} random rounds now [/cyan]",
        )
        prev_best = 10
        for rounds in range(n_rounds):
            # set seed and do a shuffle
            # random.seed(rounds*25)
            shuffled_list = list(range(0, cons_ints.shape[1]))
            random.shuffle(shuffled_list)
            for i, idx in enumerate(shuffled_list):
                # iterate over each attack
                for attack in range(cons_ints.size(0)):
                    est_int = running_intersection.copy()
                    est_union = running_union.copy()

                    # Update statistics when using `attack` for the current image, add the current attack stats and subtract the previous selected attacks numbers  # noqa: E501
                    update_int = (
                        cons_ints[attack, idx, :]
                        - cons_ints[selected_attk_indx[idx], idx, :]
                    )
                    update_union = (
                        cons_unions[attack, idx, :]
                        - cons_unions[selected_attk_indx[idx], idx, :]
                    )

                    # return est_miou and upodated running intersections and unions
                    est_miou, new_ints, new_unis = _compute_miou_subtraction(
                        torch.tensor(est_int),
                        torch.tensor(est_union),
                        torch.tensor(update_int),
                        torch.tensor(update_union),
                    )

                    # update selected attack and running values
                    if est_miou < final_miou:
                        selected_attk_indx[idx] = attack
                        running_intersection = new_ints
                        running_union = new_unis
                final_miou = _compute_miou(
                    torch.tensor(running_intersection),
                    torch.tensor(running_union),
                )

            if prev_best - final_miou <= 1e-6:
                break
            else:
                prev_best = final_miou
            final_miou = _compute_miou(
                torch.tensor(running_intersection), torch.tensor(running_union)
            )

        self.saveDict["seed"] = SEED
        self.saveDict["final_miou"] = final_miou

        # uncomment to save which attack was best image-wise for mIoU
        # self.saveDict["attack_idx"] = selected_attk_indx

        print("SEA Evaluation complete, saved-dict:")
        print(self.saveDict)

        del cons_ints
        del cons_unions
        del running_intersection
        del running_union
        del data_loader

    def worse_case_eval(self, bs=16, n_batches=-1):
        """Compute worse case aACC across 3-losses for SEA."""

        if not self.l_output:
            # load from saved-argmaxes
            self.l_output = []
            for loss_ in self.los_pairs:
                self.l_output.append(
                    torch.load(
                        self.saveDir + f"/argmax-logs/{self.modelName}_{loss_}_{self.eps}.pt"
                    )
                )

        final_acc_1 = None

        class_wise_logits = torch.stack(self.l_output)

        aaacc = []

        data_loader = self.get_loader(bs=bs)

        for i, vals in enumerate(data_loader):
            _, target = vals[0], vals[1]
            BS = target.shape[0]
            acc_cls = torch.zeros(class_wise_logits.shape[0], BS, self.n_cls)
            n_pxl_cls = torch.zeros(class_wise_logits.shape[0], BS, self.n_cls)

            pred = class_wise_logits[:, i * BS : i * BS + BS]

            acc_curr = pred == target
            intersection_all = pred == target

            for cl in range(self.n_cls):
                ind = target == cl
                ind = ind.expand(class_wise_logits.shape[0], -1, -1, -1)
                acc_curr_ = acc_curr * ind
                acc_cls[:, :, cl] += (
                    acc_curr_.view(acc_curr.shape[0], acc_curr.shape[1], -1)
                    .float()
                    .sum(2)
                )
                n_pxl_cls[:, :, cl] += (
                    ind.view(ind.shape[0], ind.shape[1], -1).float().sum(2)
                )

                intersection_all * ind

            tenss = acc_cls.sum(2) / n_pxl_cls.sum(2)
            aaacc.append(tenss)

            if i + 1 == n_batches:
                print("enough batches seen")
                break

        final_acc_1 = torch.cat(aaacc, dim=-1)

        worse_1 = final_acc_1.min(0)[0].mean()
        at_w_sum1 = final_acc_1.mean(-1)

        print("SEA evaluated Acc", worse_1)

        self.saveDict["worst_Acc"] = worse_1.item()
        self.saveDict["worst_Acc_indiv"] = at_w_sum1

        # uncomment to save the loss wise image wise aACC matrix
        # self.saveDict["final_matrix_Acc"] = final_acc_1

        del final_acc_1
        del class_wise_logits
        del acc_cls
        del n_pxl_cls
        del data_loader
