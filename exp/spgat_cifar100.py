from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.gat import SPGAT
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import CIFAR100
from src.utils.training.image_classification import spgat_batch_processing_fn

# to be changed for each experiment
save_path = "./data/models/spgat_cifar100"
data_root = "./data/image/CIFAR100"
project = "SPGAT-CIFAR100"
description = "SPGAT on CIFAR100"
batch_processing_fn = spgat_batch_processing_fn
n_segments = 16
compactness = 15
resize_stack_patches = (8, 8)
def model_init_fn(args):
    return SPGAT(
        args.channel_size,
        args.hidden_size,
        args.classes,
        args.num_heads,
        args.num_gat_layers,
        args.num_mlp_layers,
        args.dropout,
        args.pe,
        args.pe_size,
        args.image_size,
        resize_stack_patches[0],
        args.sp_agg,
    )

# experiment arguments
extra_args = [
    ("--save-path", str, save_path, "path to save the model"),
    ("--data-root", str, data_root, "path to save the dataset"),
    
    ("--image-size", int, 32, ""),
    ("--channel-size", int, 3, ""),
    ("--classes", int, 100, ""),
    
    ("--hidden-size", int, 256, ""),
    ("--num-heads", int, 8, ""),
    ("--num-gat-layers", int, 3, ""),
    ("--num-mlp-layers", int, 2, ""),
    ("--dropout", float, 0.2, ""),
    ("--pe", int, 0, ""),
    ("--pe-size", int, 256, ""),
    ("--sp-agg", str, "linear", "linear | lrgb | cnn"),
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
if exp.args.sp_agg == "linear":
    dataset.to_slic_graphs(resize_stack_patches=resize_stack_patches, n_segments=n_segments, compactness=compactness)
elif exp.args.sp_agg == "lrgb":
    dataset.to_slic_graphs(n_segments=n_segments, compactness=compactness, lrgb_stats=True)
else:
    raise NotImplementedError()
exp.prepare_dataset(dataset, graph_loader=True)

# experiment run
exp.run(model_init_fn=model_init_fn, batch_processing_fn=batch_processing_fn)