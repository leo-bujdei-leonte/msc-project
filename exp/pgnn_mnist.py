# TODO

from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.pgnn import PGNN
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import MNIST
from src.utils.training.image_classification import gcn_batch_processing_fn

# experiment arguments
extra_args = [
    ("--save-path", str, "./data/models/gcn_mnist", "path to save the model"),
    ("--data-root", str, "./data/image/MNIST",      "path to save the dataset"),
    
    ("--laplacian-pe",  int, 0,   "add laplacian eigenvector positional encodings"),
    ("--laplacian-k",   int, 100, "number of non-trivial positional eigenvectors"),
    ("--laplacian-undirected", int, 1,   "whether graph is undirected for positional encodings"),
    
    ("--image-size",      int, 28,  "image size"),
    ("--channel-size",    int, 1,   "channel size"),
    ("--classes",         int, 10,  "number of classes"),
    ("--num-conv-layers", int, 3,   "number of graph convolutional layers"),
    ("--num-lin-layers",  int, 2,   "number of mlp layers"),
    ("--hidden-size",     int, 256, "hidden size"),
]
exp = Experiment("GCN-MNIST", "GCN on MNIST")
exp.parse_args(extra_args)

# data preprocessing
transform = Compose([
    Resize((exp.args.image_size, exp.args.image_size)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = MNIST(root=exp.args.data_root, download=True, transform=transform)
dataset.to_pixel_graphs(
    laplacian_pe=exp.args.laplacian_pe,
    k=exp.args.laplacian_k,
    is_undirected=exp.args.laplacian_undirected
)
exp.prepare_dataset(dataset, graph_loader=True)

# experiment run
exp.run(
    lambda args: GCN(
        args.num_conv_layers,
        args.channel_size,
        args.hidden_size,
        args.classes,
        args.num_lin_layers,
    ),
    batch_processing_fn=gcn_batch_processing_fn,
)