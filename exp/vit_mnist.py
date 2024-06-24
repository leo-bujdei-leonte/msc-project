from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, Conv2d

from torchvision.models import ResNet
from torchvision.datasets import MNIST
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.utils.training.common import device
from src.utils.training.image_classification import train_test_loop
from src.utils.plotting import plot
from src.models.vit import ViT

save_path = "./data/models/resnet_mnist/"
        
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = MNIST(root="./data/image/MNIST", download=True, transform=transform)
train_dataset, test_dataset = random_split(dataset, [.9, .1])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# model = MyViT((1, 224, 224), n_patches=14, n_blocks=12, hidden_d=768, n_heads=12, out_d=10).to(device)

image_size = 224
channel_size = 1
patch_size = 16
embed_size = 768
num_heads = 12
classes = 10
num_layers = 12
hidden_size = 768
dropout = 0.2
model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size, dropout=dropout).to(device)

optimizer = Adam(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

num_epochs = 50

train_accs, train_losses, test_accs, test_losses = train_test_loop(model, optimizer, criterion, train_loader, test_loader, num_epochs)