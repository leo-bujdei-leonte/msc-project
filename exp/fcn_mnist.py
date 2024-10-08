from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.fcn import FCN
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import MNIST

# experiment arguments
extra_args = [
    ("--save-path", str, "./data/models/fcn_mnist", "path to save the model"),
    ("--data-root", str, "./data/image/MNIST", "path to save the dataset"),
    
    ("--image-size",      int, 28,  "image size"),
    ("--channel-size",    int, 1,   "channel size"),
    ("--classes",         int, 10, "number of classes"),
    ("--num-conv-layers", int, 3,   "number of convolutional layers"),
    ("--num-lin-layers",  int, 2,   "number of mlp layers"),
    ("--hidden-size",     int, 256, "hidden size"),
]
exp = Experiment("FCN-MNIST", "FCN on MNIST")
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
def fcn_init_fn(args):
    return FCN(
        args.num_conv_layers,
        args.channel_size,
        args.hidden_size,
        args.classes,
        args.num_lin_layers,
    )
exp.run(fcn_init_fn)