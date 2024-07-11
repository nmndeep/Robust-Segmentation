import argparse
import os
import sys

import torch

sys.path.append(os.getcwd())
import datetime
import time
from functools import partial

import yaml
from tabulate import tabulate
from torch import distributed as dist
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from val import evaluate

import semseg.utils.attacker as attacker
from semseg.datasets import *
from semseg.datasets.dataset_wrappers import *
from semseg.losses import get_loss
from semseg.models import *
from semseg.models.segmenter import create_segmenter
from semseg.optimizers import get_optimizer
from semseg.schedulers import get_scheduler
from semseg.utils.utils import *
from semseg.utils.utils import Logger, load_config_segmenter, makedir
from tools.val import Pgd_Attack

torch.backends.cudnn.deterministic = True

from timm import optim


def cycle(iterable):
    while True:
        yield from iterable


def sizeof_fmt(num, suffix="Flops"):
    for unit in ["", "Ki", "Mi", "G", "T"]:
        if abs(num) < 1000.0:
            return f"{num:3.3f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Yi{suffix}"


def create_optimizer(opt_args, model):
    return optim.create_optimizer(opt_args, model)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class Trainer:
    def __init__(self, gpu, cfg):
        # self.gpu = gpu
        # dist.init_process_group(backend="nccl")
        # torch.cuda.head_node(int(os.environ["LOCAL_RANK"]))
        # self.gpu = int(os.environ["LOCAL_RANK"])

        self.train_cfg, self.eval_cfg = cfg["TRAIN"], cfg["EVAL"]
        self.dataset_cfg, self.model_cfg = cfg["DATASET"], cfg["MODEL"]
        self.loss_cfg, self.optim_cfg, self.sched_cfg = (
            cfg["LOSS"],
            cfg["OPTIMIZER"],
            cfg["SCHEDULER"],
        )
        self.epochs, self.lr = self.train_cfg["EPOCHS"], self.optim_cfg["LR"]
        self.bs = self.train_cfg["BATCH_SIZE"]
        self.adversarial_train = self.train_cfg["ADVERSARIAL"]

        self.world_size = torch.cuda.device_count()
        if self.train_cfg["DDP"]:
            self.setup_distributed()

        self.ignore_label = -1  # redundant

        if self.model_cfg["NAME"] == "SegMenter":
            if self.model_cfg["BACKBONE"] == "vit_base_patch16_SAM":
                model_cfg, dataset_cfg = load_config_segmenter(
                    self.model_cfg["BACKBONE"], self.dataset_cfg["N_CLS"]
                )
                self.model = create_segmenter(
                    model_cfg,
                    self.model_cfg["PRETRAINED"],
                    self.model_cfg["BACKBONE"],
                )

            else:
                model_cfg, dataset_cfg = load_config_segmenter(
                    self.model_cfg["BACKBONE"], self.dataset_cfg["N_CLS"]
                )
                self.model = create_segmenter(model_cfg, self.model_cfg["PRETRAINED"])
        else:
            # TODO Make this consistent, remove hardcoding of values for vit-s

            model_cfg, dataset_cfg = load_config_segmenter(
                self.model_cfg["BACKBONE"], self.dataset_cfg["N_CLS"]
            )
            self.model = create_segmenter(model_cfg, self.model_cfg["PRETRAINED"])

        self.attack = self.train_cfg["ATTACK"]
        self.model = self.model.to(self.gpu)

        self.indi_head = self.train_cfg["INDI_HEAD"]  # finetune individual-heads?
        self.attack = self.train_cfg["ATTACK"]


        self.model = self.model.to(self.gpu)

        self.save_dir = str(cfg["SAVE_DIR"])

        if self.gpu == 0:
            self.save_path = (
                f"{self.save_dir}/"
                + str(self.model_cfg["NAME"])
                + "_"
                + str(self.model_cfg["BACKBONE"])
                + f'_adv_{self.adversarial_train}_{str(datetime.datetime.now())[:-7].replace(" ", "-").replace(":", "_")}'
                + str(cfg["ADDENDUM"])
            )
            makedir(self.save_path)

            self.logger = Logger(self.save_path + "/train_log")

        self.train_loader, self.val_loader = self.dataloaders()

        if self.rank == 0:
            print("No. of GPUS:", torch.cuda.device_count())
            self.logger.log(str(cfg))
            self.logger.log(str(self.model))

        self.init_optim_log()
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.gpu],
            find_unused_parameters=True,
            broadcast_buffers=False,
        )


    def setup_distributed(
        self,
    ): 
        # DDP setting
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        self.distributed = self.world_size > 1
        # ngpus_per_node = torch.cuda.device_count()
        self.local_rank = -1
        if self.local_rank != -1:  # for torch.distributed.launch
            self.rank = self.local_rank
            self.gpu = self.local_rank
        elif "SLURM_PROCID" in os.environ:  # for slurm scheduler
            self.rank = int(os.environ["SLURM_PROCID"])
            self.gpu = self.rank % torch.cuda.device_count()
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )

    def cleanup_distributed(self):
        dist.destroy_process_group()

    def init_optim_log(self):
        model = self.model

        self.loss_fn = get_loss(self.loss_cfg["NAME"], self.ignore_label, None)

        self.optimizer = get_optimizer(
            model,
            self.optim_cfg["NAME"],
            self.lr,
            self.optim_cfg["WEIGHT_DECAY"],
            self.dataset_cfg["NAME"].lower(),
            str(self.model_cfg["BACKBONE"]),
        )

        self.scheduler = get_scheduler(
            self.sched_cfg["NAME"],
            self.optimizer,
            self.epochs * self.iters_per_epoch,
            self.sched_cfg["POWER"],
            self.iters_per_epoch * self.sched_cfg["WARMUP"],
            self.sched_cfg["WARMUP_RATIO"],
        )



    def dataloaders(self):
        input_transform = transforms.Compose([transforms.ToTensor()])
        # dataset and dataloader
        data_kwargs = {
            "transform": input_transform,
            "base_size": self.train_cfg["BASE_SIZE"],
            "crop_size": self.train_cfg["IMAGE_SIZE"],
        }
        train_dataset = get_segmentation_dataset(
            self.dataset_cfg["NAME"],
            root=self.dataset_cfg["ROOT"],
            split="train",
            mode="train",
            **data_kwargs,
        )

        data_kwargs = {
            "transform": input_transform,
            "base_size": self.eval_cfg["BASE_SIZE"],
            "crop_size": self.eval_cfg["IMAGE_SIZE"],
        }

        val_dataset = get_segmentation_dataset(
            self.dataset_cfg["NAME"],
            root=self.dataset_cfg["ROOT"],
            split="val",
            mode="val",
            **data_kwargs,
        )
        self.iters_per_epoch = len(train_dataset) // (self.world_size * self.bs)
        self.max_iters = self.epochs * self.iters_per_epoch
        workers = 8  # num cpu per task

        self.train_sampler = make_data_sampler(
            train_dataset, shuffle=True, distributed=self.train_cfg["DDP"]
        )
        train_batch_sampler = make_batch_data_sampler(
            self.train_sampler, self.bs, self.max_iters
        )
        val_sampler = make_data_sampler(val_dataset, False, self.train_cfg["DDP"])
        val_batch_sampler = make_batch_data_sampler(val_sampler, self.bs)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=workers,
            pin_memory=False,
        )

        return train_loader, val_loader

    def main(self):
        model = self.model

        # for epoch in range(self.epochs):
        time1 = time.time()
        model.train()

        if self.adversarial_train:
            if self.attack == "pgd":
                attack = Pgd_Attack(
                    num_iter=self.train_cfg["N_ITERS"],
                    epsilon=self.train_cfg["EPS"] / 255.0,
                    alpha=1e-2,
                    los=self.train_cfg["LOSS_FN"],
                )
                attack_fn = partial(attack.adv_attack)
            else:
                attack_fn = partial(
                    attacker.apgd_train,
                    norm="Linf",
                    eps=self.train_cfg["EPS"] / 255.0,
                    n_iter=self.train_cfg["N_ITERS"],
                    use_rs=True,
                    loss="ce-avg",
                    is_train=False,
                    verbose=False,
                    track_loss=None,
                    logger=None,
                    gpuu=self.gpu,
                )

        criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        best_miou = 0.0
        for iterr, (img, lbl) in enumerate(self.train_loader):
            if (iterr + 1) % self.iters_per_epoch:
                train_loss = 0.0

            if iterr <= 5 and self.rank == 0:
                print(img.min(), img.max())
                if self.adversarial_train:
                    self.logger.log(
                        f"{self.attack}-for {self.train_cfg['N_ITERS']} iterations  for eps: {self.train_cfg['EPS']}/255, alpha : 1e-2"
                    )
            self.optimizer.zero_grad(set_to_none=True)

            img = img.cuda(self.gpu, non_blocking=True)
            lbl = lbl.cuda(self.gpu, non_blocking=True)

            with autocast(enabled=self.train_cfg["AMP"]):
                # TODO fix this properly for SEGMENTER
                if self.adversarial_train:
                    model.eval()
                    img = attack_fn(model, img, lbl)[0]
                    model.train()
                if self.model_cfg["NAME"] != "SegMenter":
                    loss, logits = model(img, lbl)
                else:
                    seg_pred = model.forward(img)
                    loss = criterion(seg_pred, lbl)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            lr = self.scheduler.get_lr()
            lr = sum(lr) / len(lr)
            train_loss += loss.item()
            eta_seconds = ((time.time() - time1) / (iterr + 1)) * (self.max_iters - iterr)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if self.rank == 0 and (iterr + 1) % (self.iters_per_epoch // 2) == 0:
                self.logger.log(
                    f"Epoch: {iterr // self.iters_per_epoch + 1:d}/{self.epochs:d} | Iters: {iterr:d}/{self.max_iters:d} || Lr: {lr:.6f} || Loss(total): {train_loss / (self.iters_per_epoch // 10):.5f} ||  Cost Time: {str(datetime.timedelta(seconds=int(time.time() - time1)))} || Estimated Time: {eta_string}"
                )
                train_loss = 0.0

            eval_freq = 2

            if self.rank == 0 and (iterr + 1) % (self.iters_per_epoch * eval_freq) == 0:
                model.eval()

                eval__stats = evaluate(
                    model,
                    self.val_loader,
                    self.gpu,
                    self.dataset_cfg["N_CLS"],
                    n_batches=30,
                )

                if eval__stats[-1] > best_miou:
                    best_miou = eval__stats[-1]
                    torch.save(
                        model.module.state_dict()
                        if self.train_cfg["DDP"]
                        else model.state_dict(),
                        self.save_path + "/best_model_ckpt.pth",
                    )
                self.logger.log(
                    f"Epoch: [{iterr // self.iters_per_epoch + 1}] , mIoU : {eval__stats[-1]:.2f}, mAcc : {eval__stats[1]:.2f}, aAcc: {eval__stats[2]:.2f}"
                )
                self.logger.log(f"Best mIoU: {best_miou:.3f}")
                torch.save(
                    model.module.state_dict()
                    if self.train_cfg["DDP"]
                    else model.state_dict(),
                    self.save_path + f"/model_ckpt_{str(iterr + 1)}.pth",
                )
                model.train()

        end = time.gmtime(time.time() - time1)
        if self.rank == 0:
            ckpt1 = torch.load(
                self.save_path + "/best_model_ckpt.pth", map_location="cpu"
            )
            ckpt = {}
            for k in ckpt1.keys():
                new_key = "module." + k
                ckpt[new_key] = ckpt1.pop(k)

            model.load_state_dict(ckpt)
            eval__stats = evaluate(
                model,
                self.val_loader,
                self.gpu,
                self.dataset_cfg["N_CLS"],
                n_batches=100,
            )
            miou = eval__stats[-1]
            aacc = eval__stats[2]
            macc = eval__stats[1]

            table = [
                ["Eval", "ADE"],
                ["Best mIoU", f"{miou:.2f}"],
                ["Best mAcc", f"{macc:.2f}"],
                ["Best aAcc", f"{aacc:.2f}"],
                ["Total training Time", time.strftime("%H:%M:%S", end)],
            ]
            print(tabulate(table, numalign="right"))
            self.logger.log(
                "Final stats: ADE - Best mIoU: {miou:.2f} - Best aAcc: {aacc: .2f}"
            )

    @classmethod
    def launch_from_args(cls, world_size, cfg):
        distributed = True
        if distributed:
            torch.multiprocessing.spawn(
                cls._exec_wrapper, args=(cfg,), nprocs=world_size, join=True
            )
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        cls.exec(*args, **kwargs)

    @classmethod
    def exec(cls, gpu, cfg):
        trainer = cls(gpu=gpu, cfg=cfg)
        trainer.main()
        trainer.cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/pascalvoc_convnext_cvst.yaml",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--world_size", type=int, default=6, help="Configuration file to use"
    )
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    trainer = Trainer(gpu=0, cfg=cfg)
    trainer.main()
    trainer.cleanup_distributed()
    # Trainer.launch_from_args(args.world_size, cfg)
