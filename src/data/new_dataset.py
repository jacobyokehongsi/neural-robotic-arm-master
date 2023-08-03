import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import zipfile
import numpy as np


class CyclicDataset(Dataset):
    def __init__(
        self,
        path: str,
        transforms: Callable
    ):
        self.X = np.load(path+'_data.npy')
        self.y = np.load(path+'_label.npy')
        self.transforms = transforms
        if self.transforms is not None:
            self.X = self.transforms(self.X)
            self.y = self.transforms(self.y)
        self.X = self.X.reshape(-1, self.X.shape[-1])
        self.y = self.y.reshape(-1, self.y.shape[-1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx, :]

    def split_data(self, split_ratio=0.1):
        val_size = round(split_ratio * len(self.X))
        train_size = len(self.X) - val_size
        return torch.utils.data.random_split(self, [train_size, val_size])


class RoboticArmDataset(LightningDataModule):
    """
    PyTorch Lightning data module 
    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        test_batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:

        train_dataset = CyclicDataset(path=self.data_dir+'_train',
                                      transforms=transforms.ToTensor())
        self.train_dataset, self.val_dataset = train_dataset.split_data(
            split_ratio=0.1)
        self.test_dataset = CyclicDataset(path=self.data_dir+'_test',
                                          transforms=transforms.ToTensor())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
