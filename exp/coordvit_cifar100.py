from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.slic_transformer import CoordViT
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import CIFAR100
from src.utils.training.image_classification import coordvit_batch_processing_fn
from src.utils.preprocess import collate_slic_graph_patches

# to be changed for each experiment
save_path = "./data/models/coordvit_cifar100"
data_root = "./data/image/CIFAR100"
project = "CoordViT-CIFAR100"
description = "CoordViT on CIFAR100"
batch_processing_fn = coordvit_batch_processing_fn
n_segments = 16
compactness = 15
resize_stack_patches = (8, 8)
def model_init_fn(args):
    return CoordViT(
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
    )

# experiment arguments
extra_args = [
    ("--save-path", str, save_path, "path to save the model"),
    ("--data-root", str, data_root, "path to save the dataset"),
    
    ("--batch-size",   int,   4096, "batch size for the dataloader"),
    ("--weight-decay", float, 0.0,  "weight decay"),
    
    ("--image-size",    int,   224,  "image size"),
    ("--channel-size",  int,   3,    "channel size"),
    ("--patch-size",    int,   16,   "patch size"),
    ("--embed-size",    int,   768,  "embed size"),
    ("--num-heads",     int,   12,   "number of heads"),
    ("--classes",       int,   100,  "number of classes"),
    ("--num-layers",    int,   12,   "number of layers"),
    ("--hidden-size",   int,   768,  "hidden size"),
    ("--dropout",       float, 0.2,  "dropout"),
    ("--mlp-size",      int,   3072, "mlp hidden size"),
]
exp = Experiment(project, description)
exp.parse_args(extra_args)

# data preprocessing
transform = Compose([
    Resize((exp.args.image_size, exp.args.image_size)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = CIFAR100(root=exp.args.data_root, download=True, transform=transform)
dataset.to_slic_graphs(image_size=exp.args.image_size, resize_stack_patches=resize_stack_patches, n_segments=n_segments, compactness=compactness)
exp.prepare_dataset(dataset, graph_loader=False, batch_collate_fn=collate_slic_graph_patches)

# experiment run
exp.run(model_init_fn=model_init_fn, batch_processing_fn=batch_processing_fn)