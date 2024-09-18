from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.gat import GAT2, GAT1
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import MNIST
from src.utils.training.image_classification import gat_batch_processing_fn

# to be changed for each experiment
save_path = "./data/models/gat_mnist"
data_root = "./data/image/MNIST"
project = "GAT-MNIST"
description = "GAT on MNIST"
batch_processing_fn = gat_batch_processing_fn
def model_init_fn(args):
    return GAT2(
        args.channel_size,
        args.hidden_size,
        args.classes,
        args.num_heads,
        args.num_gat_layers,
        args.num_mlp_layers,
        args.dropout,
    )

# experiment arguments
extra_args = [
    ("--save-path", str, save_path, "path to save the model"),
    ("--data-root", str, data_root, "path to save the dataset"),
    
    ("--laplacian-pe",  int, 0,   "add laplacian eigenvector positional encodings"),
    ("--laplacian-k",   int, 256, "number of non-trivial positional eigenvectors"),
    ("--laplacian-undirected", int, 1,   "whether graph is undirected for positional encodings"),
    ("--precompute-laplacian", int, 0, "compute PE before batching"),
    
    ("--sinusoidal-pe", int, 0, "add sinusoidal PE"),
    
    ("--image-size",      int, 28,  "image size"),
    ("--channel-size",    int, 1,   "channel size"),
    ("--classes",         int, 10,  "number of classes"),
    ("--num-gat-layers", int, 3,   "number of graph attention layers"),
    ("--num-mlp-layers",  int, 2,   "number of mlp layers"),
    ("--num-heads",  int, 8,   "number of attention heads"),
    ("--hidden-size",     int, 256, "hidden size"),
    ("--dropout",       float, 0.2, "dropout"),
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
if exp.args.precompute_laplacian:
    dataset.to_pixel_graphs(
        laplacian_pe=exp.args.laplacian_pe,
        k=exp.args.laplacian_k,
        is_undirected=exp.args.laplacian_undirected
    )
else:
    dataset.to_pixel_graphs()
exp.prepare_dataset(dataset, graph_loader=True)

# experiment run
exp.run(model_init_fn=model_init_fn, batch_processing_fn=batch_processing_fn)