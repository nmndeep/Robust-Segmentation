import torch
import argparse
import yaml
import math, random
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
from semseg.datasets.dataset_wrappers import *
import torch.utils.data as data
from torchvision import transforms
from rich.console import Console
import numpy as np
from matplotlib import pyplot as plt
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from PIL import Image, ImageDraw, ImageFont

from semseg.utils.visualize import generate_palette
console = Console()
SEED = 225


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(SEED)
    random.seed(SEED)

g = torch.Generator()
g.manual_seed(SEED)


class SemSeg:
    def __init__(self, cfg, save_dir) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = torch.tensor(np.loadtxt('/scratch/nsingh/sem_seg/assests/ade/ade20k_colors.txt').astype('uint8'))
        d = get_segmentation_dataset(cfg['DATASET']['NAME'], root=cfg['DATASET']['ROOT'])
        self.labels = d.classes
        # print(self.labels)
        # initialize the model and load weights and send to device
        # self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'])
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], cfg['DATASET']['N_CLS'], None)
        self.model.load_state_dict(torch.load(cfg['EVAL']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.save_dir = save_dir
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

    def postprocess(self, image, label, seg_map):
        # resize to original image size
        # seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # get segmentation map (value being 0 to num_classes)
        print(f"Image Shape\t: {image.shape}")
        print(f"Label Shape\t: {label.shape}")
        print(f"Seg-map Shape\t: {seg_map.shape}")
        print(f"Classes\t\t: {label.unique().tolist()}")
        palette = torch.tensor(generate_palette(150))
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
    
        plt.imshow(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
        plt.savefig(str(self.save_dir) + f"/seed_{SEED}_output_image.png")

        
    def predict(self, val_data) -> Tensor:
        

        dataloader = DataLoader(val_data, shuffle=True, batch_size=4,     worker_init_fn=seed_worker, generator=g)
        image, label, _ = next(iter(dataloader))

        _, seg_map = self.model(pixel_values=image.to(self.device), labels=label.to(self.device))
        self.postprocess(image.cpu(), label.cpu(), seg_map)


def get_val_data(dataset_cfg):

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': 520, 'crop_size': 512}

    val_dataset = get_segmentation_dataset(dataset_cfg['NAME'], root=dataset_cfg['ROOT'], split='val', mode='val', **data_kwargs)

    return val_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k_convnext_vena.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    dataset_cfg, model_cfg, test_cfg = cfg['DATASET'], cfg['MODEL'], cfg['EVAL']

    test_data = get_val_data(dataset_cfg)

    # if not test_file.exists():
    #     raise FileNotFoundError(test_file)

    console.print(f"Model > [yellow]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/yellow]")
    console.print(f"Dataset > [yellow]{cfg['DATASET']['NAME']}[/yellow]")

    save_dir = Path(cfg['SAVE_DIR']) / 'test_results'
    save_dir.mkdir(exist_ok=True)
    
    semseg = SemSeg(cfg, save_dir)
    with console.status("[bright_green]Processing..."):
        segmap = semseg.predict(test_data)

        # if test_file.is_file():
        #     console.rule(f'[green]{test_file}')
        #     segmap.save(save_dir / f"{str(test_file.stem)}.png")
        # else:
        #     files = test_file.glob('*.*')
        #     for file in files:
        #         console.rule(f'[green]{file}')
        #         segmap = semseg.predict(str(file), cfg['TEST']['OVERLAY'])
        #         segmap.save(save_dir / f"{str(file.stem)}.png")

    console.rule(f"[cyan]Segmentation results are saved in `{save_dir}`")