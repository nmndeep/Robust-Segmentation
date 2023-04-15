from .ade import ADE20KSegmentation
from .camvid import CamVid
from .cityscapes import CityScapes
from .cocostuff import COCOStuff
from .cihp import CIHP, CCIHP
from .suim import SUIM
from .celebamaskhq import CelebAMaskHQ
from .distributed_sampler import DistributedSampler, IterationBasedBatchSampler
from .pascal_aug import VOCAugSegmentation
from .pascal_voc import VOCSegmentation
import torch.utils.data as data

from torchvision import transforms

from torch import distributed as dist

__all__ = [
    'CamVid',
    'CityScapes',
    'ADE20KSegmentation',
    'CIHP',
    'CCIHP',
    'COCOStuff',
    'SUIM',
    'CelebAMaskHQ',
    'IterationBasedBatchSampler',
    'DistributedSampler',
    'get_segmentation_dataset',
    'make_data_sampler', 
    'make_batch_data_sampler',
    'VOCAugSegmentation',
    'VOCSegmentation'
]

datasets= {'ade20k': ADE20KSegmentation,    
'pascalvoc': VOCSegmentation,
'pascalaug': VOCAugSegmentation}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)


def make_data_sampler(dataset, shuffle, distributed=True):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return sampler

def make_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0):
    batch_sampler = data.sampler.BatchSampler(sampler, images_per_batch, drop_last=True)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler
