"""
Contains classes to interact with torchvision datasets.
"""
import os
from enum import Enum

import numpy as np
import torch.utils.data as torchdatautils
import torchvision.datasets as datasets
from .tiny_imagenet import TinyImageNet


class DatasetEnum(Enum):
    """
    Enum class holding dataset name mappings to lowercase values to be used internally.
    """
    MNIST = ("mnist")
    CIFAR10 = ("cifar10")
    CIFAR100 = ("cifar100")
    SVHN = ("svhn")
    ImageNet = ("imagenet")
    OMNIGLOT = ("omniglot")
    TIM = ("tim")
    LSUN = ("lsun")

    def __init__(self, name):
        self._value_ = name

    @classmethod
    def getEnum(cls, name):
        return cls._member_map_.get(name)

class BaseData:
    """
    Cutsom wrapper on any torch vision dataset to download all available datasets -
    train, val, test - whichever is supported.
    """
    def __init__(self):
        self.train_args = {
            DatasetEnum.MNIST: {'train': True},
            DatasetEnum.CIFAR10: {'train': True},
            DatasetEnum.CIFAR100: {'train': True},
            DatasetEnum.SVHN: {'split': 'train'},
            DatasetEnum.ImageNet: {'split': 'train'},
            DatasetEnum.OMNIGLOT: {'background': True}
        } 
        self.val_args = {

        }
        self.test_args = {
            DatasetEnum.MNIST: {'train': False},
            DatasetEnum.CIFAR10: {'train': False},
            DatasetEnum.CIFAR100: {'train': False},
            DatasetEnum.SVHN: {'split': 'test'},
            DatasetEnum.ImageNet: {'split': 'val'},
            DatasetEnum.OMNIGLOT: {'background': False},
            DatasetEnum.TIM: {'split': 'test'},
            DatasetEnum.LSUN: {'classes': 'test'}
        }
        self.data = {}

    def loadData(self, data_dir, transform, target_transform, dataset, dataset_builder):
        assert isinstance(dataset, DatasetEnum)
        assert dataset_builder is not None

        # load train dataset if one available
        if dataset in self.train_args:
            self.data['train'] = dataset_builder(root=data_dir,
                                                 transform=transform,
                                                 target_transform=target_transform,
                                                 download=True,
                                                 **self.train_args[dataset])

        # load val dataset if one available
        if dataset in self.val_args:
            self.data['val'] = dataset_builder(root=data_dir,
                                               transform=transform,
                                               target_transform=target_transform,
                                               download=True,
                                               **self.val_args[dataset])

        # load test dataset if one available
        if dataset in self.test_args:
            self.data['test'] = dataset_builder(root=data_dir,
                                                transform=transform,
                                                target_transform=target_transform,
                                                download=True,
                                                **self.test_args[dataset])

class TorchVisionDataWrapper:
    """
    Dataset class for any of the standard torchvision datasets available.
    Provides a common interface to access all datasets and manitains a train, val, test
    dataset if directly provided by the dataset or splits the train dataset into train,
    val based on the input val ratio specified.

    Args:
        dataset: str - should be one of the supported datasets.
    """
    def __init__(self):
        self.supported_datasets = DatasetEnum.__dict__.keys()
        self.dataset_builders = {
            DatasetEnum.MNIST: datasets.MNIST,
            DatasetEnum.CIFAR10: datasets.CIFAR10,
            DatasetEnum.CIFAR100: datasets.CIFAR100,
            DatasetEnum.SVHN: datasets.SVHN,
            DatasetEnum.ImageNet: datasets.ImageNet,
            DatasetEnum.OMNIGLOT: datasets.Omniglot,
            DatasetEnum.TIM: TinyImageNet,
            DatasetEnum.LSUN: datasets.LSUN
        }

    def get_dataset(self, dataset, data_dir,
                    transform, target_transform, split_type: str,
                    val_ratio=None, test_ratio=None):
        assert split_type in ['train', 'test']
        assert dataset in self.supported_datasets

        # load dataset
        basedata = BaseData()
        dataset_enum = DatasetEnum.getEnum(dataset)
        basedata.loadData(os.path.join(data_dir, dataset_enum.value),
                          transform, target_transform, dataset_enum,
                          self.dataset_builders[dataset_enum])
        if split_type == 'train':
            num_train = len(basedata.data['train'])
            train_indices = np.arange(num_train)
            # np.random.shuffle(train_indices)

            if val_ratio is not None:
                # make sure we have train, val datasets if val_ratio is present
                num_val = int(val_ratio * num_train)
                num_train = int((1-val_ratio) * num_train)
                trainset = torchdatautils.Subset(basedata.data['train'],
                                                 train_indices[:num_train])
                valset = torchdatautils.Subset(basedata.data['train'],
                                               train_indices[num_train:(num_train + num_val)])
                return trainset, valset
            else:
                # make sure we have train dataset if val_ratio is not present
                return basedata.data['train']
        elif split_type == 'test':
            if basedata.data['test'] is not None:
                return basedata.data['test']

if __name__ == "__main__":
    vis = TorchVisionDataWrapper()
    vis.get_dataset('MNIST', './data', None, None, 'train', val_ratio=0.1)
