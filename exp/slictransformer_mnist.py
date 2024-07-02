import os
import pickle
from time import time

import torch
from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.utils.training.common import device
from src.models.slic_transformer import SLICTransformer
from src.utils.training.image_classification import train_test_loop

save_path = "./data/models/slictransformer_mnist/"

dataset = pickle.load(open("./data/image/MNIST/MNIST_SLIC_avg_224_196_0p1.pkl", "rb"))
max_len = max(d[0].shape[0] for d in dataset)
for i in range(len(dataset)):
    x = torch.zeros(max_len, dataset[0][0].shape[1])
    x[:dataset[i][0].shape[0]] = dataset[i][0]
    dataset[i] = (x, dataset[i][1])
print("Finished preprocessing")

def collate_fn(batch): # TODO mask
    return tuple(zip(*batch))

train_dataset, test_dataset = random_split(dataset, [.9, .1])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)#, collate_fn=collate_fn)

in_size = 3
embed_size = 512
num_heads = 8
classes = 10
num_layers = 3
hidden_size = 256
dropout = 0.2
model = SLICTransformer(in_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout).to(device)

optimizer = Adam(model.parameters(), lr=5e-4)
criterion = CrossEntropyLoss()

num_epochs = 50

metrics = train_test_loop(
    model,
    optimizer,
    criterion,
    train_loader, 
    test_loader, 
    num_epochs,
    save_path=save_path,
    plot=True,
)