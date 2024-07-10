import pickle

from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from src.utils.data.preprocess import resize_stack_slic_graph_patches, collate_slic_graph_patches
from src.models.slic_transformer import CoordViT
from src.utils.training.common import device
from src.utils.training.image_classification import train_test_loop, train_epoch_coordvit, eval_coordvit

save_path = "./data/models/coordvit_mnist/"

dataset = pickle.load(open("./data/image/MNIST/MNIST_SLIC_graph_28_16_0p5.pkl", "rb"))
dataset = resize_stack_slic_graph_patches(dataset, (7, 7)) # this is part of the model
print("Finished preprocessing")

train_dataset, test_dataset = random_split(dataset, [.9, .1])
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_slic_graph_patches
)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_slic_graph_patches
)

image_size = 28
channel_size = 1
patch_size = 7
embed_size = 512
num_heads = 8
classes = 10
num_layers = 3
hidden_size = 256
dropout = 0.2
model = CoordViT(
    image_size,
    channel_size,
    patch_size,
    embed_size,
    num_heads,
    classes,
    num_layers,
    hidden_size,
    dropout=dropout
).to(device)


optimizer = Adam(model.parameters(), lr=5e-5)
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
    train_epoch_fn=train_epoch_coordvit,
    eval_fn=eval_coordvit,
)