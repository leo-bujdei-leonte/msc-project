from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.enoughvit import EnoughViT
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import MNIST
from src.utils.training.image_classification import default_batch_processing_fn

# to be changed for each experiment
save_path = "./data/models/enoughvit_mnist"
data_root = "./data/image/MNIST"
project = "EnoughViT-MNIST"
description = "EnoughViT on MNIST"
batch_processing_fn = default_batch_processing_fn
def model_init_fn(args):
    return EnoughViT(
        args.image_size,
        args.channel_size,
        args.patch_size,
        args.embed_size,
        args.num_heads,
        args.classes,
        args.num_layers,
        args.hidden_size,
        args.mlp_size,
        dropout=args.dropout,
        learnable_pe=args.learnable_pe,
    )

# experiment arguments
extra_args = [
    ("--save-path", str, save_path, "path to save the model"),
    ("--data-root", str, data_root, "path to save the dataset"),
    
    ("--batch-size",   int,   32, "batch size for the dataloader"),
    ("--weight-decay", float, 0.0,  "weight decay"),
    
    ("--image-size",    int,   28,  "image size"),
    ("--channel-size",  int,   1,    "channel size"),
    ("--patch-size",    int,   7,   "patch size"),
    ("--embed-size",    int,   512,  "embed size"),
    ("--num-heads",     int,   8,   "number of heads"),
    ("--classes",       int,   10,  "number of classes"),
    ("--num-layers",    int,   3,   "number of layers"),
    ("--hidden-size",   int,   256,  "hidden size"),
    ("--dropout",       float, 0.2,  "dropout"),
    ("--learnable-pe",  int,   0,    "learnable positional encoding"),
    ("--mlp-size",      int,   256, "mlp hidden size"),
]
exp = Experiment(project, description)
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
exp.run(model_init_fn=model_init_fn, batch_processing_fn=batch_processing_fn)