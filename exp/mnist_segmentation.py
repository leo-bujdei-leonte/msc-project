import pickle
import os

from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.datasets import MNIST

from src.utils.segmentation import segment_image_slic_avg_pool, image_to_SLIC_graph

data_path = "./data/image/MNIST"
      
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = MNIST(root=data_path, download=True, transform=transform)

os.makedirs(data_path, exist_ok=True)

# SLIC superpixels with mean pool and centroid position
# file_name = "/MNIST_SLIC_avg_224_196_0p1.pkl"
# if not os.path.isfile(data_path+file_name):
#     segmented_dataset = []
#     for idx, (x, y) in enumerate(dataset):
#         segmented_dataset.append((segment_image_slic_avg_pool(x, 196, 0.1), y))
#         if idx % 1000 == 0:
#             print("Processed image", idx+1)

#     pickle.dump(segmented_dataset, open(data_path+file_name, "wb"))

# file_name = "/MNIST_SLIC_avg_224_196_0p5.pkl"
# if not os.path.isfile(data_path+file_name):
#     segmented_dataset = []
#     for idx, (x, y) in enumerate(dataset):
#         segmented_dataset.append((segment_image_slic_avg_pool(x, 196, 0.5), y))
#         if idx % 1000 == 0:
#             print("Processed image", idx+1)

#     pickle.dump(segmented_dataset, open(data_path+file_name, "wb"))
    
# SLIC graph with patches and centroid position
# file_name = "/MNIST_SLIC_graph_224_196_0p5.pkl"
# if not os.path.isfile(data_path+file_name):
#     segmented_dataset = []
#     for idx, (x, y) in enumerate(dataset):
#         g = image_to_SLIC_graph(x, n_segments=196, compactness=0.5)
#         g.y = y
#         segmented_dataset.append(g)
#         if idx % 1000 == 0:
#             print("Processed image", idx+1)
#             if os.path.isfile(data_path+file_name):
#                 os.remove(data_path+file_name)
#             pickle.dump(segmented_dataset, open(data_path+file_name, "wb"))

#     pickle.dump(segmented_dataset, open(data_path+file_name, "wb"))

transform = Compose([
    Resize((28, 28)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = MNIST(root=data_path, download=True, transform=transform)

file_name = "/MNIST_SLIC_graph_28_16_0p5.pkl"
if not os.path.isfile(data_path+file_name):
    segmented_dataset = []
    for idx, (x, y) in enumerate(dataset):
        g = image_to_SLIC_graph(x, n_segments=16, compactness=0.5)
        g.y = y
        segmented_dataset.append(g)
        if idx % 1000 == 0:
            print("Processed image", idx+1)
            if os.path.isfile(data_path+file_name):
                os.remove(data_path+file_name)
            pickle.dump(segmented_dataset, open(data_path+file_name, "wb"))

    pickle.dump(segmented_dataset, open(data_path+file_name, "wb"))