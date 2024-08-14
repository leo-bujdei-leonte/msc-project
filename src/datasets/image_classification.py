import os
import pickle

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.utils.preprocess import random_split, image_to_pygraph, resize_stack_slic_graph_patches
from src.utils.segmentation import image_to_SLIC_graph

class ImageClassificationDataset(Dataset):
    def __init__(self, root, *args, **kwargs) -> None:
        super().__init__()
        self.root = root
        self.data = []
        self.train_data, self.val_data, self.test_data = [], [], []
        self.train_loader, self.val_loader, self.test_loader = None, None, None
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    def to_pixel_graphs(self):
        g_path = os.sep.join([self.root, "graph", "pixel.pkl"])
        if os.path.isfile(g_path):
            self.data = pickle.load(open(g_path, "rb"))
            print("Loaded existing pixel graphs")
        
        else:
            data = []
            for idx, datum in enumerate(self.data):
                data.append(image_to_pygraph(datum))
                if idx % 1000 == 0:
                    print("Processed image", idx)
            os.makedirs(os.path.dirname(g_path), exist_ok=True)
            pickle.dump(data, open(g_path, "wb"))
            print("Converted image dataset to pixel graphs")
            
            self.data = data
    
    def _convert_to_slic(self, n_segments, compactness):
        g_path = os.sep.join([self.root, "graph", f"SLIC_{n_segments}_{compactness}.pkl"])
        if os.path.isfile(g_path):
            data = pickle.load(open(g_path, "rb"))
            print("Loaded existing SLIC graphs")
        
        else:
            data = []
            for idx, datum in enumerate(self.data):
                data.append(image_to_SLIC_graph(datum, n_segments, compactness))
                if idx % 1000 == 0:
                    print("Processed image", idx)
            os.makedirs(os.path.dirname(g_path), exist_ok=True)
            pickle.dump(data, open(g_path, "wb"))    
            print("Converted image dataset to SLIC graphs")
        
        return data
    
    def to_slic_graphs(self, n_segments, compactness, resize_stack_patches=None):
        if resize_stack_patches is not None:
            g_path = os.sep.join([self.root, "graph", f"SLIC_{n_segments}_{compactness}_resized_stacked_{resize_stack_patches}.pkl"])
            if os.path.isfile(g_path):
                self.data = pickle.load(open(g_path, "rb"))
                print("Loaded existing resized and stacked graphs")
            
            else:
                self.data = self._convert_to_slic(n_segments, compactness)
                
                print("Resizing and stacking patches")
                self.data = resize_stack_slic_graph_patches(self.data, resize_stack_patches)
                os.makedirs(os.path.dirname(g_path), exist_ok=True)
                pickle.dump(self.data, open(g_path, "wb"))
        
        else:
            self.data = self._convert_to_slic(n_segments, compactness)
    
    def splits(self, ratios):
        assert sum(ratios) == 1

        if self.train_data == []:
            self.train_data, self.val_data, self.test_data = random_split(self.data, ratios, "CIFAR100")
        return self.train_data, self.val_data, self.test_data

    def loaders(self, batch_size, shuffles=(True, False, False), num_workers=0, graph_loader=False, collate_fn=None):
        LoaderClass = PyGDataLoader if graph_loader else DataLoader
        if self.train_loader is None:
            self.train_loader = LoaderClass(
                self.train_data,
                batch_size=batch_size,
                shuffle=shuffles[0],
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            self.val_loader = LoaderClass(
                self.val_data,
                batch_size=batch_size,
                shuffle=shuffles[1],
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
            self.test_loader = LoaderClass(
                self.test_data,
                batch_size=batch_size,
                shuffle=shuffles[2],
                num_workers=num_workers,
                collate_fn=collate_fn,
            )
        return self.train_loader, self.val_loader, self.test_loader
    
class CIFAR100(ImageClassificationDataset):
    def __init__(self, root, transform=None, download=True) -> None:
        super().__init__(root=root)
        
        self.data = datasets.CIFAR100(root, transform=transform, download=download)


class MNIST(ImageClassificationDataset):
    def __init__(self, root, transform=None, download=True) -> None:
        super().__init__(root=root)
        
        self.data = datasets.MNIST(root, transform=transform, download=download)
        
        
        
if __name__ == "__main__":
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize
    
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(0, 1),
    ])
    dataset = MNIST(root="./data/image/MNIST", download=True, transform=transform)
    train_data, val_data, test_data = dataset.splits([0.8, 0.1, 0.1])
    train_loader, val_loader, test_loader = dataset.loaders(32)
    
    print(dataset[0][0].shape)
    