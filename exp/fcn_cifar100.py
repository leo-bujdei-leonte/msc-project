from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.fcn import FCN
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import CIFAR100

# experiment arguments
extra_args = [
    ("--save-path", str, "./data/models/fcn_cifar100", "path to save the model"),
    ("--data-root", str, "./data/image/CIFAR100", "path to save the dataset"),
    
    ("--image-size",      int, 32,  "image size"),
    ("--channel-size",    int, 3,   "channel size"),
    ("--classes",         int, 100, "number of classes"),
    ("--num-conv-layers", int, 3,   "number of convolutional layers"),
    ("--num-lin-layers",  int, 3,   "number of mlp layers"),
    ("--hidden-size",     int, 256, "hidden size"),
]
exp = Experiment("FCN-cifar100", "FCN on CIFAR100")
exp.parse_args(extra_args)

# data preprocessing
transform = Compose([
    Resize((exp.args.image_size, exp.args.image_size)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = CIFAR100(root=exp.args.data_root, download=True, transform=transform)
exp.prepare_dataset(dataset)

# experiment run
def vit_init_fn(args):
    return FCN(
        args.num_conv_layers,
        args.channel_size,
        args.hidden_size,
        args.classes,
        args.num_lin_layers,
    )
exp.run(vit_init_fn)