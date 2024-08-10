from torch.utils.data import Dataset, DataLoader

from torchvision import datasets

from src.utils.preprocess import random_split

class ImageClassificationDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.data = []
        self.train_data, self.val_data, self.test_data = [], [], []
        self.train_loader, self.val_loader, self.test_loader = None, None, None
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def splits(self, ratios):
        assert sum(ratios) == 1

        if self.train_data == []:
            self.train_data, self.val_data, self.test_data = random_split(self.data, ratios, "CIFAR100")
        return self.train_data, self.val_data, self.test_data

    def loaders(self, batch_size, shuffles=(True, False, False), num_workers=0, collate_fn=None):
        if self.train_loader is None:
            self.train_loader = DataLoader(
                self.train_data,
                batch_size=batch_size,
                shuffle=shuffles[0],
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            self.val_loader = DataLoader(
                self.val_data,
                batch_size=batch_size,
                shuffle=shuffles[1],
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            self.test_loader = DataLoader(
                self.test_data,
                batch_size=batch_size,
                shuffle=shuffles[2],
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
        return self.train_loader, self.val_loader, self.test_loader
    
class CIFAR100(ImageClassificationDataset):
    def __init__(self, root, transform=None, download=True) -> None:
        super().__init__()
        
        self.data = datasets.CIFAR100(root, transform=transform, download=download)
        
        
if __name__ == "__main__":
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(0, 1),
    ])
    dataset = CIFAR100(root="./data/image/CIFAR100", download=True)
    train_data, val_data, test_data = dataset.splits([0.8, 0.1, 0.1])
    train_loader, val_loader, test_loader = dataset.loaders(4096)
    
    print(len(train_data), len(test_data), len(val_data), len(dataset))
    print(len(train_loader), len(test_loader), len(val_loader), len(dataset))
    