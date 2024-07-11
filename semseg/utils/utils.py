import argparse
import os
import random
import shutil
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch import distributed as dist
from torch.backends import cudnn

ADE_WTS = [
    2.5511e-05,
    3.6983e-05,
    4.7570e-05,
    6.6522e-05,
    1.1128e-04,
    1.0635e-04,
    1.6683e-04,
    1.7398e-04,
    2.2595e-04,
    2.6104e-04,
    3.4525e-04,
    3.2680e-04,
    4.6724e-04,
    2.7333e-04,
    3.8664e-04,
    5.1101e-04,
    4.6147e-04,
    2.8707e-04,
    4.8777e-04,
    5.6734e-04,
    5.2263e-04,
    5.7571e-04,
    7.9284e-04,
    7.1656e-04,
    9.8619e-04,
    7.3393e-04,
    6.0752e-04,
    6.0696e-04,
    1.0648e-03,
    1.5916e-03,
    7.4704e-04,
    1.3956e-03,
    1.0427e-03,
    1.6245e-03,
    1.3812e-03,
    1.2803e-03,
    1.5659e-03,
    2.3384e-03,
    2.6498e-03,
    2.1948e-03,
    1.9984e-03,
    2.1434e-03,
    2.2654e-03,
    2.3339e-03,
    2.6016e-03,
    2.9368e-03,
    2.4439e-03,
    2.5844e-03,
    2.3346e-03,
    1.0170e-03,
    2.7078e-03,
    3.7222e-03,
    3.0739e-03,
    3.0697e-03,
    5.0181e-03,
    4.7774e-03,
    2.0477e-03,
    3.1477e-03,
    2.8421e-03,
    3.7206e-03,
    2.5296e-03,
    2.1699e-03,
    2.8066e-03,
    2.8080e-03,
    5.5795e-03,
    4.0186e-03,
    4.8758e-03,
    3.5471e-03,
    3.1513e-03,
    3.0316e-03,
    3.9002e-03,
    5.0847e-03,
    4.8401e-03,
    5.9311e-03,
    5.3158e-03,
    5.0188e-03,
    4.0362e-03,
    4.4585e-03,
    5.2076e-03,
    4.4833e-03,
    5.5491e-03,
    5.7523e-03,
    5.5545e-03,
    8.7588e-03,
    5.0301e-03,
    5.4497e-03,
    7.6726e-03,
    5.1451e-03,
    7.9943e-03,
    4.4696e-03,
    7.4416e-03,
    6.7389e-03,
    7.8750e-03,
    5.5496e-03,
    1.2515e-02,
    5.1635e-03,
    8.1806e-03,
    9.9495e-03,
    1.0522e-02,
    6.0337e-03,
    1.1848e-02,
    1.0531e-02,
    6.0837e-03,
    8.0876e-03,
    1.1750e-02,
    8.2409e-03,
    6.8528e-03,
    8.1382e-03,
    8.7929e-03,
    7.6437e-03,
    5.7786e-03,
    1.3009e-02,
    1.8844e-02,
    1.0949e-02,
    4.2059e-03,
    5.7906e-03,
    1.2998e-02,
    1.4171e-02,
    7.0287e-03,
    9.0963e-03,
    1.0115e-02,
    1.0510e-02,
    1.3813e-02,
    1.2319e-02,
    1.4154e-02,
    1.5693e-02,
    1.5035e-02,
    1.1120e-02,
    1.6888e-02,
    7.3436e-03,
    1.4521e-02,
    9.3029e-03,
    1.4782e-02,
    1.1918e-02,
    1.7509e-02,
    2.0762e-02,
    1.4547e-02,
    2.0312e-02,
    1.0543e-02,
    1.8876e-02,
    3.6659e-02,
    2.0046e-02,
    2.2035e-02,
    1.4011e-02,
    1.5645e-02,
    1.1985e-02,
    1.0001e-02,
    2.7073e-02,
    2.1668e-02,
    1.8419e-02,
    2.1877e-02,
]

VOC_WTS = [
    0.0007,
    0.0531,
    0.1394,
    0.0500,
    0.0814,
    0.0575,
    0.0256,
    0.0312,
    0.0198,
    0.0626,
    0.0382,
    0.0457,
    0.0212,
    0.0404,
    0.0421,
    0.0089,
    0.0915,
    0.0585,
    0.0366,
    0.0279,
    0.0677,
]


class ImageNormalizer(nn.Module):
    def __init__(
        self, mean: tuple[float, float, float], std: tuple[float, float, float]
    ) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer(
            "mean", torch.as_tensor(mean).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.as_tensor(std).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


def normalize_model(
    model: nn.Module,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> nn.Module:
    layers = OrderedDict([("normalize", ImageNormalizer(mean, std)), ("model", model)])
    return nn.Sequential(layers)


def make_attack_dirs(saveloc):
    save_dir = Path(saveloc) / "test_results"
    save_dir.mkdir(exist_ok=True)
    save_dir = Path(saveloc) / "sea-stats"
    save_dir.mkdir(exist_ok=True)
    save_dir = Path(saveloc) / "argmax-logs"
    save_dir.mkdir(exist_ok=True)

def remove_dirs(saveloc):
    try:
        shutil.rmtree(Path(saveloc) / "test_results")
        shutil.rmtree(Path(saveloc) / "argmax-logs")
    except:
        print("Couldn't delete intermediate files")

def writeIndivloss(saveloc, modelName, clean_stats, test_eps, loss_, adv_stats):
    with open(
        saveloc + f"/sea-stats/loss_wise_{modelName}_{loss_}_N_{test_eps}.txt",
        "a+",
    ) as f:
        f.write(f"{modelName} \n")
        f.write(f"Clean stats: {clean_stats}\n")
        f.write(f"----- Linf radius: {test_eps} ------")
        f.write(f"Attack: {loss_} \n")
        f.write(f"Adversarial results: {adv_stats}\n")


def getModelName(mname, backname):
    if mname == "SegMenter":
        modelname = "SegMent_" + backname
    elif mname == "UperNetForSemanticSegmentation":
        modelname = "UperNet_" + backname
    else:
        modelname = "PSPNet_RN50"
    return modelname


def load_config_segmenter(backbone, n_cls):
    cfg1 = yaml.load(open("./configs/segmenter.yml"), Loader=yaml.FullLoader)
    model_cfg = cfg1["model"][f"{backbone}"]
    dataset_cfg = cfg1["dataset"]["ade20k"]
    decoder_cfg = cfg1["decoder"]["mask_transformer"]

    im_size = 512
    crop_size = dataset_cfg.get("crop_size", im_size)
    window_size = dataset_cfg.get("window_size", im_size)

    window_stride = dataset_cfg.get("window_stride", im_size)

    model_cfg["image_size"] = (crop_size, crop_size)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = 0.0
    model_cfg["drop_path_rate"] = 0.1
    decoder_cfg["name"] = "mask_transformer"
    model_cfg["decoder"] = decoder_cfg
    model_cfg["n_cls"] = n_cls

    return model_cfg, dataset_cfg


def optim_args_segmenter(bs, epochs):
    # only works for ADE20K
    optimizer_kwargs = dict(
        opt="sgd",
        lr=0.001,
        weight_decay=0.00001,
        momentum=0.9,
        clip_grad=None,
        sched="polynomial",
        epochs=epochs,
        min_lr=1e-5,
        poly_power=0.9,
        poly_step_size=1,
    )
    # optimizer
    optimizer_kwargs["iter_max"] = (25574 // bs) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v

    return opt_args


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path + ".txt"

    def log(self, str_to_log):
        print(str_to_log)
        if self.log_path is not None:
            with open(self.log_path, "a") as f:
                f.write(str_to_log + "\n")
                f.flush()


def fix_seeds(seed: int = 3407) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def setup_cudnn() -> None:
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    cudnn.benchmark = True
    cudnn.deterministic = False


def time_sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def get_model_size(model: nn.Module | torch.jit.ScriptModule):
    tmp_model_path = Path("temp.p")
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, tmp_model_path)
    else:
        torch.save(model.state_dict(), tmp_model_path)
    size = tmp_model_path.stat().st_size
    os.remove(tmp_model_path)
    return size / 1e6  # in MB


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6  # in M


def setup_ddp() -> int:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(gpu)
        dist.init_process_group("nccl", world_size=world_size, rank=rank)
        dist.barrier()
    else:
        gpu = 0
    return gpu


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: Tensor) -> Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
