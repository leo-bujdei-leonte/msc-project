from torch.nn import Conv2d

from torchvision.models import ResNet
from torchvision.datasets import MNIST
from torchvision.models.resnet import BasicBlock
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.experiments.image_classification import Experiment
from src.datasets.image_classification import MNIST

# experiment arguments
extra_args = [
    ("--save-path", str, "./data/models/resnet_mnist", "path to save the model"),
    ("--data-root", str, "./data/image/MNIST", "path to save the dataset"),
    
    ("--image-size",      int, 224,  "image size"),
    ("--channel-size",    int, 3,   "channel size"),
    ("--classes",         int, 10, "number of classes"),
]
exp = Experiment("ResNet-MNIST", "ResNet on MNIST")
exp.parse_args(extra_args)

# data preprocessing
transform = Compose([
    Resize((exp.args.image_size, exp.args.image_size)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = MNIST(root=exp.args.data_root, download=True, transform=transform)
exp.prepare_dataset(dataset)

# experiment run
class MNISTResNet(ResNet):
    def __init__(self) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
def resnet_init_fn(args):
    return MNISTResNet()
exp.run(resnet_init_fn)