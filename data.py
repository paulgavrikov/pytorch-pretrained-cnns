from abc import ABC

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, EMNIST, KMNIST, Omniglot, FashionMNIST


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.num_classes = 10
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.root_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.root_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR100Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)
        self.num_classes = 10
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.root_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR100(root=self.root_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class OmniglotData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0,)
        self.std = (1,)
        self.num_classes = 1623
        self.in_channels = 1

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = Omniglot(root=self.root_dir, background=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = Omniglot(root=self.root_dir, background=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class TensorData(pl.LightningDataModule):
    def __init__(self, data_class, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = data_class

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class MNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(MNIST, root_dir, batch_size, num_workers)
        self.mean = (0.1307,)
        self.std = (0.3081,)
        self.num_classes = 10
        self.in_channels = 1


class KMNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(KMNIST, root_dir, batch_size, num_workers)
        self.mean = (0,)
        self.std = (1,)
        self.num_classes = 49
        self.in_channels = 1


class FashionMNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(FashionMNIST, root_dir, batch_size, num_workers)
        self.mean = (0,)
        self.std = (1,)
        self.num_classes = 10
        self.in_channels = 1


all_datasets = {
    "cifar10": CIFAR10Data,
    "cifar100": CIFAR100Data,
    "mnist": MNISTData,
    "kmnist": KMNISTData,
    "omniglot": OmniglotData,
    "fashionmnist": FashionMNISTData
}


def get_dataset(name):
    return all_datasets.get(name)
