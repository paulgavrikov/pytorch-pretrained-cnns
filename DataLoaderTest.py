import os
import unittest

import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from data import GroceryStore, HistAerial25x25Data, HistAerial50x50Data, HistAerial100x100Data, TinyImageNetData, \
    TinyImageNet, FractalDB60Data


class MyTestCase(unittest.TestCase):
    def test_grocery_dataset(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        test_o = GroceryStore("G:/Users/David/Source/Repos/MasterThesisRepos/GroceryStoreDataset",
                              split="train", transform=transform)
        object_get = test_o.__getitem__(2)
        print(object_get)
        # object_get[0].close()

    def test_grocery_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = GroceryStore(root_dir="G:/Users/David/Source/Repos/MasterThesisRepos/GroceryStoreDataset",
                               split="train", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=8,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        for data in dataloader:
            print(data)

    def test_tinyimagenet_dataloader_class(self):
        hist_data = TinyImageNetData(root_dir=os.path.join("C:/DataSets/", "tinyimagenet"),
                                        batch_size=256, num_workers=8)
        train_loader = hist_data.train_dataloader()
        val_loader = hist_data.val_dataloader()

        print(train_loader)

        # for item in train_loader:
        #     print(item)

    def test_tinyimagenet_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = TinyImageNet(root_dir=os.path.join("C:/DataSets/", "tinyimagenet"), mode="train", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=8,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # for data in dataloader:
        #     print(data)

    def test_hist_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(100),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ImageFolder(root=os.path.join("C:/DataSets/test", "100x100_overlap_0percent"), transform=transform)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.15 * num_train))

        train_idx = indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=8,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )

        for data in dataloader:
            print(data)

    def test_fractaldb_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ImageFolder(root=os.path.join("C:/DataSets", "fractaldb60"),
                              transform=transform)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.15 * num_train))

        train_idx = indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=8,
            sampler=train_sampler,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )

        for data in dataloader:
            print(data)

    def test_fractaldb_dataloader_class(self):
        hist_data = FractalDB60Data(root_dir=os.path.join("C:/DataSets", "fractaldb60"),
                                        batch_size=256, num_workers=8)
        train_loader = hist_data.train_dataloader()
        val_loader = hist_data.val_dataloader()

        for item in train_loader:
            print(item)

    def test_hist_dataloader_25x25_class(self):
        hist_data = HistAerial25x25Data(root_dir=os.path.join("C:/DataSets/test", "25x25_overlap_0percent"),
                                        batch_size=256, num_workers=8)
        train_loader = hist_data.train_dataloader()
        val_loader = hist_data.val_dataloader()

        # for item in train_loader:
        #     print(item)

    def test_hist_dataloader_50x50_class(self):
        hist_data = HistAerial50x50Data(root_dir=os.path.join("C:/DataSets/test", "50x50_overlap_0percent"),
                                        batch_size=256, num_workers=8)
        train_loader = hist_data.train_dataloader()
        val_loader = hist_data.val_dataloader()

    def test_hist_dataloader_100x100_class(self):
        hist_data = HistAerial100x100Data(root_dir=os.path.join("C:/DataSets/test", "100x100_overlap_0percent"),
                                          batch_size=256, num_workers=8)
        train_loader = hist_data.train_dataloader()
        val_loader = hist_data.val_dataloader()


if __name__ == '__main__':
    unittest.main()
