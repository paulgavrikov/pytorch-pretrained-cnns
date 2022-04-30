import json
import os

import hub
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, KMNIST, FashionMNIST, ImageFolder, SVHN


class ImageNet1k(Dataset):
    def __init__(self, data_root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}

        with open(os.path.join(data_root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        samples_dir = os.path.join(data_root, split)
        for syn_id in os.listdir(samples_dir):
            target = self.syn_to_class[syn_id]
            syn_folder = os.path.join(samples_dir, syn_id)
            for sample in os.listdir(syn_folder):
                sample_path = os.path.join(syn_folder, sample)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx])
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class ImageNet1kData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = 1000
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageNet1k(root=self.root_dir, split="train", transform=transform)
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
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageNet1k(root=self.root_dir, split="val", transform=transform)
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


class GroceryStore(Dataset):
    def __init__(self, root, split="train", transform=None):
        self.root_dir = root
        self.samples_frame = []
        self.transform = transform

        if split not in ['train', 'val', 'test']:
            raise ValueError(f"slit value has to be one of {['train', 'val', 'test']}")
        dataset_path = None

        if split == "train":
            dataset_path = "train.txt"
        if split == "val":
            dataset_path = "val.txt"
        if split == "test":
            dataset_path = "test.txt"

        with open(os.path.join(root, dataset_path), "rb") as f:
            self.samples_frame = pd.read_csv(f)

    def __len__(self):
        return len(self.samples_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.samples_frame.iloc[idx, 0])
        x = Image.open(img_name)
        if self.transform:
            x = self.transform(x)
        return x, self.samples_frame.iloc[idx, 2]


class GroceryStoreData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.num_classes = 42
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = GroceryStore(root=self.root_dir, split="train", transform=transform)
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
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = GroceryStore(root=self.root_dir, split="val", transform=transform)
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


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.49139968, 0.48215841, 0.44653091)
        self.std = (0.24703223, 0.24348513, 0.26158784)
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


class TinyImageNetData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.num_classes = 200
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        ds = hub.load("hub://activeloop/tiny-imagenet-train")
        dataloader = ds.pytorch(
            transform={'images': transform, 'labels': None},
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
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        ds = hub.load("hub://activeloop/tiny-imagenet-validation")
        dataloader = ds.pytorch(
            transform={'images': transform, 'labels': None},
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        ds = hub.load("hub://activeloop/tiny-imagenet-test")
        dataloader = ds.pytorch(
            transform={'images': transform, 'labels': None},
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader


class SVHNData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
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
        dataset = SVHN(root=self.root_dir, split="train", transform=transform, download=True)
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
        dataset = SVHN(root=self.root_dir, split="test", transform=transform, download=True)
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
        self.mean = (0.50707516, 0.48654887, 0.44091784)
        self.std = (0.26733429, 0.25643846, 0.27615047)
        self.num_classes = 100
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
        dataset = CIFAR100(root=self.root_dir, train=True, transform=transform, download=True)
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


class CINIC10Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers, part="all"):
        super().__init__()
        assert part in ["all", "imagenet", "cifar10"]
        self.part = part
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.47889522, 0.47227842, 0.43047404)  # from https://github.com/BayesWatch/cinic-10
        self.std = (0.24205776, 0.23828046, 0.25874835)
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
        dataset = ImageFolder(root=os.path.join(self.root_dir, "train"), transform=transform, is_valid_file= \
            lambda path: (self.part == "all") or \
                         (self.part == "imagenet" and not os.path.basename(path).startswith("cifar10-")) or \
                         (self.part == "cifar10" and os.path.basename(path).startswith("cifar10-")))
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
        dataset = ImageFolder(root=os.path.join(self.root_dir, "valid"), transform=transform)
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
                transforms.Resize((32, 32)),
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
                transforms.Resize((32, 32)),
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
        self.mean = (0.1918,)
        self.std = (0.3483,)
        self.num_classes = 49
        self.in_channels = 1


class FashionMNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(FashionMNIST, root_dir, batch_size, num_workers)
        self.mean = (0.2860,)
        self.std = (0.3530,)
        self.num_classes = 10
        self.in_channels = 1


all_datasets = {
    "cifar10": CIFAR10Data,
    "cifar100": CIFAR100Data,
    "mnist": MNISTData,
    "kmnist": KMNISTData,
    "fashionmnist": FashionMNISTData,
    "cinic10": CINIC10Data,
    "imagenet1k": ImageNet1kData,
    "svhn": SVHNData,
    "tinyimagenet": TinyImageNetData
}


def get_dataset(name):
    return all_datasets.get(name)
