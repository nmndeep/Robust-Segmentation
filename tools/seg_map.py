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





class SemSeg:
    def __init__(self, cfg, save_dir, adv=False) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], cfg['DATASET']['N_CLS'], None)
        self.model_name = str(cfg['MODEL']['NAME'])+ '--' + str(cfg['MODEL']['BACKBONE'])
        self.model.load_state_dict(torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu'))
        self.dataset_name = str(cfg['DATASET']['NAME'])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.save_dir = save_dir
        self.adversarial = adv
        self.n_cls = cfg['DATASET']['N_CLS']
        # preprocess parameters and transformation pipeline
        self.size = cfg['EVAL']['IMAGE_SIZE']
        self.tf_pipeline = transforms.Compose([
            transforms.Lambda(lambda x: x / 255),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = transforms.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, image, label, seg_map, n_cls):
        # resize to original image size
        # seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # get segmentation map (value being 0 to num_classes)
        print(f"Image Shape\t: {image.shape}")
        print(f"Label Shape\t: {label.shape}")
        print(f"Seg-map Shape\t: {seg_map.shape}")
        print(f"Classes\t\t: {label.unique().tolist()}")
        palette = torch.tensor(generate_palette(n_cls))
        label[label == -1] = 0
        label[label == 255] = 0
        labels = [palette[lbl.to(int)].permute(2, 0, 1) for lbl in label]
        labels = torch.stack(labels)
        seg_map[seg_map == -1] = 0
        seg_map[seg_map == 255] = 0
        seg_maps = [palette[lbl.to(int)].permute(2, 0, 1) for lbl in seg_map]
        seg_maps = torch.stack(seg_maps)

        inv_normalize = transforms.Normalize(
            mean=(-0.485/0.229, -0.456/0.224, -0.406/0.225),
            std=(1/0.229, 1/0.224, 1/0.225)
        )
        image = inv_normalize(image)
        image *= 255
        images = torch.vstack([image, labels, seg_maps])
        labels = ['Image', 'Ground-truth', 'Output-map']
        f, axarr = plt.subplots(3,4, figsize=(15, 10), sharex=True, sharey=True)
        for i in range(3):
            idx = 0
            for j in range(4):
                axarr[i,j].imshow(images[i*4+idx].to(torch.uint8).numpy().transpose(1, 2, 0))
                axarr[i,j].set_xticks([])
                axarr[i,j].set_yticks([])
                idx+=1
        for i in range(len(labels)):
            plt.setp(axarr[i, 0], ylabel=labels[i])
        plt.suptitle(f"{self.dataset_name} output maps of {self.model_name}")
        # plt.imshow(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
        ax = plt.gca()

        # Hide X and Y axes label marks
        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(labelleft=False)

        # Hide X and Y axes tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig(str(self.save_dir) + f"/output_images/seed_{SEED}_adv_{self.adversarial}_{self.model_name}_data_{n_cls}_output_image.png", dpi=300)

        
    def predict(self, val_data) -> Tensor:
        
        batch_size = 4
        if not self.adversarial:
            dataloader = DataLoader(val_data, shuffle=True, batch_size=batch_size,     worker_init_fn=seed_worker, generator=g)
            image, label, _ = next(iter(dataloader))
        else:
            image = val_data['images'][:batch_size]
            label = val_data['labels'][:batch_size]
        self.model.eval()
        seg_map = self.model(pixel_values=image.to(self.device), labels=label.to(self.device))
        self.postprocess(image.cpu(), label.cpu(), seg_map, self.n_cls)


def get_data(dataset_cfg, test_cfg):

    if str(test_cfg['NAME']) == 'pascalvoc':
        data_dir = '../VOCdevkit/'
        val_data = get_segmentation_dataset(test_cfg['NAME'],
            root=dataset_cfg['ROOT'],
            split='val',
            transform=torchvision.transforms.ToTensor(),
            base_size=512,
            crop_size=(473, 473))

    elif str(test_cfg['NAME']) == 'pascalaug':
        val_data = get_segmentation_dataset(test_cfg['NAME'],
            root=dataset_cfg['ROOT'],
            split='val',
            transform=torchvision.transforms.ToTensor(),
            base_size=512,
            crop_size=(473, 473))

    elif str(test_cfg['NAME']).lower() == 'ade20k':
        val_data = get_segmentation_dataset(test_cfg['NAME'],
            root=dataset_cfg['ROOT'],
            split='val',
            transform=torchvision.transforms.ToTensor(),
            base_size=520,
            crop_size=(512, 512))
    else:
        raise ValueError(f'Unknown dataset.')


    # val_loader = torch.utils.data.DataLoader(
    #     val_data, batch_size=48, shuffle=False,
    #     num_workers=1, pin_memory=True, sampler=None, worker_init_fn =seed_worker, generator=g)

    return val_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/pascalvoc_cvst_clean.yaml')
    parser.add_argument('--adversarial-data', action='store_true', help='PGD data?', default=False)

    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    dataset_cfg, model_cfg, test_cfg = cfg['DATASET'], cfg['MODEL'], cfg['EVAL']

    if not args.adversarial_data:
        val_data = get_val_data(dataset_cfg, test_cfg)
    else:
        val_data = torch.load(cfg['SAVE_DIR'] + f"/test_results/adv_data_{str(cfg['MODEL']['BACKBONE'])}.pt")

    console.print(f"Model > [yellow]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/yellow]")
    console.print(f"Dataset > [yellow]{cfg['DATASET']['NAME']}[/yellow]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)



    semseg = SemSeg(cfg, save_dir, args.adversarial_data)
    with console.status("[bright_green]Processing..."):
        segmap = semseg.predict(val_data)

     
    console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")