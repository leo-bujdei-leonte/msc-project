from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.utils.training.common import device
from src.utils.training.image_classification import train_test_loop
from src.models.vit import ViT

save_path = "./data/models/vit_mnist/"
        
transform = Compose([
    Resize((28, 28)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = MNIST(root="./data/image/MNIST", download=True, transform=transform)
train_dataset, test_dataset = random_split(dataset, [.9, .1])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

image_size = 28
channel_size = 1
patch_size = 7
embed_size = 512
num_heads = 8
classes = 10
num_layers = 3
hidden_size = 256
dropout = 0.2
model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout=dropout, learnable_pe=False).to(device)

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
    save_path=None,
    plot=False,
)