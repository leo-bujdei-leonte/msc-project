from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.gcn import GCN
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import CIFAR100
from src.utils.training.image_classification import gcn_batch_processing_fn

# to be changed for each experiment
save_path = "./data/models/gcn_cifar100"
data_root = "./data/image/CIFAR100"
project = "GCN-cifar100"
description = "GCN on CIFAR100"
batch_processing_fn = gcn_batch_processing_fn
def model_init_fn(args):
    if not args.precompute_laplacian:
        return GCN(
            args.num_conv_layers,
            args.channel_size,
            args.hidden_size,
            args.classes,
            args.num_lin_layers,
            args.laplacian_pe,
            args.laplacian_k,
            args.laplacian_undirected
        )
    else:
        return GCN(
            args.num_conv_layers,
            args.channel_size + args.laplacian_k,
            args.hidden_size,
            args.classes,
            args.num_lin_layers,
            False,
            args.laplacian_k,
            args.laplacian_undirected
        )

# experiment arguments
extra_args = [
    ("--save-path", str, save_path, "path to save the model"),
    ("--data-root", str, data_root, "path to save the dataset"),
        
    ("--laplacian-pe",         int, 0,   "add laplacian eigenvector positional encodings"),
    ("--laplacian-k",          int, 256, "number of non-trivial positional eigenvectors"),
    ("--laplacian-undirected", int, 1,   "whether graph is undirected for positional encodings"),
    ("--precompute-laplacian", int, 0,   "compute PE before batching"),
    
    ("--image-size",      int, 32,  "image size"),
    ("--channel-size",    int, 3,   "channel size"),
    ("--classes",         int, 100, "number of classes"),
    ("--num-conv-layers", int, 3,   "number of convolutional layers"),
    ("--num-lin-layers",  int, 3,   "number of mlp layers"),
    ("--hidden-size",     int, 256, "hidden size"),
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
exp.run(
    model_init_fn=model_init_fn,
    batch_processing_fn=gcn_batch_processing_fn,
)