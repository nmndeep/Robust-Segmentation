import torch.utils.data as data

from .ade import ADE20KSegmentation
from .distributed_sampler import (
    DistributedSampler,
    IterationBasedBatchSampler,
)
from .pascal_aug import VOCAugSegmentation
from .pascal_voc import VOCSegmentation

__all__ = [
    "ADE20KSegmentation",
    "IterationBasedBatchSampler",
    "DistributedSampler",
    "get_segmentation_dataset",
    "make_data_sampler",
    "make_batch_data_sampler",
    "VOCAugSegmentation",
    "VOCSegmentation",
]

datasets = {
    "ade20k": ADE20KSegmentation,
    "pascalvoc": VOCSegmentation,
    "pascalaug": VOCAugSegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)


def make_data_sampler(dataset, shuffle, distributed=True):
    if distributed:
        return DistributedSampler(dataset, shuffle=shuffle)
    # return sampler


def make_batch_data_sampler(sampler, images_per_batch, num_iters=None, start_iter=0):
    batch_sampler = data.sampler.BatchSampler(sampler, images_per_batch, drop_last=True)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler
