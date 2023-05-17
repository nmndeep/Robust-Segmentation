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
from collections import OrderedDict

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tools.val import Pgd_Attack, clean_accuracy
from PIL import Image, ImageDraw, ImageFont
import gc
from autoattack.other_utils import check_imgs
import torch.nn as nn
from functools import partial
from semseg.utils.visualize import generate_palette
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer, create_optimizers, adjust_learning_rate
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, Logger, makedir, normalize_model
from val import evaluate, Pgd_Attack
import semseg.utils.attacker as attacker
import semseg.datasets.transform_util as transform
from semseg.metrics import Metrics
import torchvision
# from mmcv.utils import Config
# from mmcv.runner import get_dist_info
from semseg.datasets.dataset_wrappers import *
console = Console()
SEED = 225
random.seed(SEED)
np.random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


BASE_DIR = '/data/naman_deep_singh/model_zoo/seg_models/test_results/output_logits_new/'
# apgd_mask-ce-avg_5iter_rob_mod_0.0471_n_it_100_pascalvoc_ConvNeXt-T_CVST_ROB_SD_220.pt


DATAS = 'pascalvoc' # 'pascalvoc'
EPS = 0.0627 #0.0157, 0.0314, 0.0471, 0.0627
ITERR = "c_init_5iter_" #, #"S_mod"
ATTACK = 'apgd'#-larg-eps'
strr = [
f"{ATTACK}_mask-ce-avg_{ITERR}_rob_mod_{EPS}_n_it_300_{DATAS}_ConvNeXt-T_CVST_ROB_SD_225_1x300.pt", 
f"{ATTACK}_segpgd-loss_{ITERR}_rob_mod_{EPS}_n_it_300_{DATAS}_ConvNeXt-T_CVST_ROB_SD_225_1x300.pt",
f"{ATTACK}_js-avg_{ITERR}_rob_mod_{EPS}_n_it_300_{DATAS}_ConvNeXt-T_CVST_ROB_SD_225_1x300.pt",
f"{ATTACK}_mask-norm-corrlog-avg_{ITERR}_rob_mod_{EPS}_n_it_300_{DATAS}_ConvNeXt-T_CVST_ROB_SD_225_1x300.pt"
]




# if ITERR == '5iter_mod':
#     fold = '5iter_rob_model'
# elif ITERR == '2iter':
#     fold = '2iter_rob_model'
# elif ITERR == 'S_mod':
#     fold = 'S_model'
# else:
# fold = 'clean_model_out'
fold = '5iter_rob_model'
# fold = '5iter_rob_model'
print("before loading pt files")
# for i in range(len(strr)):
    # tens1 = torch.load(BASE_DIR + f"{fold}/preds/" + strr[i])
    # print(tens1.size())
    # tenss = tens1.max(1)[1]
    # torch.save(tenss, BASE_DIR + f"{fold}/preds/" + strr[i][:-3]+"_MAX.pt")
tenss = (torch.load(BASE_DIR + f"{fold}/worse_cases/WORST_CASE_apgd_0.0627_5iter_mod_pascalvoc_ConvNeXt-T_CVST_ROB_SD_225_1x300.pt")) # + strr[i]))
print(tenss.keys())
print("miou worse", tenss["worse_miou_across_4_losses"])
print(tenss["pair_wise_key"])
print(tenss["pair_wise_Acc"])
print(tenss["worst_Acc_across_4_losses"])

    # l_output2.append(torch.load(BASE_DIR + f"{fold}/" +strr[i][:-3]+ "_SD_220.pt"))
    # print(l_output[-1].size())
# exit()