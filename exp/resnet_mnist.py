from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, Conv2d

from torchvision.models import ResNet
from torchvision.datasets import MNIST
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.utils.training.common import device
from src.utils.training.image_classification import train_test_loop

save_path = "./data/models/resnet_mnist/"

class MNISTResNet(ResNet):
    def __init__(self) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = MNIST(root="./data/image/MNIST", download=True, transform=transform) # TOOD should not reset each time
train_dataset, test_dataset = random_split(dataset, [.9, .1])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = MNISTResNet().to(device)

optimizer = Adam(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()

num_epochs = 15

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

train_losses, train_accs, test_losses, test_accs = metrics
epochs = list(range(1, num_epochs+1))