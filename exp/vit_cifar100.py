from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.vit import ViT
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import CIFAR100

# experiment arguments
extra_args = [
    ("--save-path",   str, "./data/models/vit_cifar100", "path to save the model"),
    ("--data-root",          str,   "./data/image/CIFAR100", "path to save the dataset"),
    
    ("--image-size",    int,   32,    "image size"),
    ("--channel-size",  int,   3,     "channel size"),
    ("--patch-size",    int,   8,     "patch size"),
    ("--embed-size",    int,   512,   "embed size"),
    ("--num-heads",     int,   8,     "number of heads"),
    ("--classes",       int,   100,   "number of classes"),
    ("--num-layers",    int,   3,     "number of layers"),
    ("--hidden-size",   int,   256,   "hidden size"),
    ("--dropout",       float, 0.2,   "dropout"),
    ("--learnable-pe",  int,   0, "learnable positional encoding"),
]
exp = Experiment("ViT-cifar100", "ViT on CIFAR100")
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
    return ViT(
        args.image_size,
        args.channel_size,
        args.patch_size,
        args.embed_size,
        args.num_heads,
        args.classes,
        args.num_layers,
        args.hidden_size,
        dropout=args.dropout,
        learnable_pe=args.learnable_pe,
    )
exp.run(vit_init_fn)