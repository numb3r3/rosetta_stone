from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import datasets, transforms

from rosetta.core.dataio import BaseDataIO

class CIFAR10(BaseDataIO):
    def __init__(self, **kwargs):
        self.cache_path = kwargs.get("cache_path", ".cache/data/cifar10")

        self.norm = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        self.da_transform = [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]

    def create_dataset(
        self,
        file_paths: List[str],
        mode: str = "train",
        download: bool = True,
        **kwargs,
    ):
        # assert not download, "Download dataset by yourself!"
        assert download or os.path.exists(self.cache_path), "cache path does not exist"

        is_train = mode == "train"
        transform = (
            transforms.Compose(self.da_transform + self.norm) if is_train else self.norm
        )
        dt = datasets.CIFAR10(
            self.cache_path, train=is_train, transform=transform, download=download
        )
        return dt

    def collate_fn(
        self, batch, tensor_names=None, mode: str = "train", **kwargs
    ) -> Dict[str, torch.Tensor]:
        images = []
        labels = []
        for img, label in batch:
            images.append(img)
            labels.append(label)
        return (torch.stack(images, dim=0), torch.tensor(labels, dtype=torch.int64))