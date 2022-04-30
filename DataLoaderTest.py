import unittest

from torch.utils.data import DataLoader
from torchvision import transforms

from data import GroceryStore


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

        test_o = GroceryStore("G:/Users/David/Source/Repos/MasterThesisRepos/GroceryStoreDataset/dataset",
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

        dataset = GroceryStore(root="G:/Users/David/Source/Repos/MasterThesisRepos/GroceryStoreDataset/dataset",
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


if __name__ == '__main__':
    unittest.main()
