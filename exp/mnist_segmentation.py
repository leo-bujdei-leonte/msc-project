import pickle
import os

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.datasets import MNIST

from src.models.slic_transformer import segment_image_slic_avg_pool

data_path = "./data/image/MNIST"
      
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = MNIST(root=data_path, download=True, transform=transform)

os.makedirs(data_path, exist_ok=True)

# SLIC superpixels with mean pool and centroid position
file_name = "/MNIST_SLIC_avg_224_196_0p1.pkl"
if not os.path.isfile(data_path+"file_name"):
    segmented_dataset = []
    for idx, (x, y) in enumerate(dataset):
        segmented_dataset.append((segment_image_slic_avg_pool(x, 196, 0.1), y))
        if idx % 1000 == 0:
            print("Processed image", idx+1)

    pickle.dump(segmented_dataset, open(data_path+file_name, "wb"))

file_name = "/MNIST_SLIC_avg_224_196_0p5.pkl"
if not os.path.isfile(data_path+"file_name"):
    segmented_dataset = []
    for idx, (x, y) in enumerate(dataset):
        segmented_dataset.append((segment_image_slic_avg_pool(x, 196, 0.5), y))
        if idx % 1000 == 0:
            print("Processed image", idx+1)

    pickle.dump(segmented_dataset, open(data_path+file_name, "wb"))