import torch
import numpy as np
import random
import time
import os
import functools
from pathlib import Path
from torch.backends import cudnn
from torch import nn, Tensor
from torch.autograd import profiler
from typing import Union
from torch import distributed as dist
from tabulate import tabulate
from semseg import models

from collections import OrderedDict
from typing import Tuple
import yaml
from timm import scheduler
from timm import optim

class ImageNormalizer(nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
        std: Tuple[float, float, float]) -> None:
        super(ImageNormalizer, self).__init__()

        self.register_buffer('mean', torch.as_tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.as_tensor(std).view(1, 3, 1, 1))

    def forward(self, input: Tensor) -> Tensor:
        return (input - self.mean) / self.std


def normalize_model(model: nn.Module, mean: Tuple[float, float, float],
    std: Tuple[float, float, float]) -> nn.Module:
    layers = OrderedDict([
        ('normalize', ImageNormalizer(mean, std)),
        ('model', model)
    ])
    return nn.Sequential(layers)


def load_config_segmenter(backbone, n_cls):
    cfg1 = yaml.load(
        open("/data/naman_deep_singh/sem_seg/configs/segmenter.yml", "r"), Loader=yaml.FullLoader
    )
    model_cfg = cfg1["model"][backbone]
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


def optim_args_segmenter(bs, epochs):
    #only works for ADE20K
    optimizer_kwargs=dict(
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
    optimizer_kwargs["iter_max"] = (25574//bs) * optimizer_kwargs["epochs"]
    optimizer_kwargs["iter_warmup"] = 0.0
    opt_args = argparse.Namespace()
    opt_vars = vars(opt_args)
    for k, v in optimizer_kwargs.items():
        opt_vars[k] = v

    return opt_args



def makedir(path):


    if not os.path.exists(path):
        os.makedirs(path)

class Logger():
    def __init__(self, log_path):
        self.log_path = log_path + '.txt'

    def log(self, str_to_log):
        print(str_to_log)
        if not self.log_path is None:
            with open(self.log_path, 'a') as f:
                f.write(str_to_log + '\n')
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

def get_model_size(model: Union[nn.Module, torch.jit.ScriptModule]):
    tmp_model_path = Path('temp.p')
    if isinstance(model, torch.jit.ScriptModule):
        torch.jit.save(model, tmp_model_path)
    else:
        torch.save(model.state_dict(), tmp_model_path)
    size = tmp_model_path.stat().st_size
    os.remove(tmp_model_path)
    return size / 1e6   # in MB

@torch.no_grad()
def test_model_latency(model: nn.Module, inputs: torch.Tensor, use_cuda: bool = False) -> float:
    with profiler.profile(use_cuda=use_cuda) as prof:
        _ = model(inputs)
    return prof.self_cpu_time_total / 1000  # ms

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6      # in M

def setup_ddp() -> int:
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group('nccl',world_size=world_size, rank=rank)
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

@torch.no_grad()
def throughput(dataloader, model: nn.Module, times: int = 30):
    model.eval()
    images, _  = next(iter(dataloader))
    images = images.cuda(non_blocking=True)
    B = images.shape[0]
    print(f"Throughput averaged with {times} times")
    start = time_sync()
    for _ in range(times):
        model(images)
    end = time_sync()

    print(f"Batch Size {B} throughput {times * B / (end - start)} images/s")


def show_models():
    model_names = models.__all__
    model_variants = [list(eval(f'models.{name.lower()}_settings').keys()) for name in model_names]

    print(tabulate({'Model Names': model_names, 'Model Variants': model_variants}, headers='keys'))


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time * 1000:.2f}ms")
        return value
    return wrapper_timer