from torch.utils.data import Dataset

class BaseClassificationDataset(Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.data = []
        self.train_data, self.testing_data = [], []
        self.train_loader, self.test_loader = [], []
    
    def __len__(self) -> int:
        return len(self.data)
    
    