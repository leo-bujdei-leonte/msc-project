from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.models.slic_transformer import CoordViT
from src.experiments.image_classification import Experiment
from src.datasets.image_classification import MNIST
from src.utils.training.image_classification import coordvit_batch_processing_fn
from src.utils.preprocess import collate_slic_graph_patches

# to be changed for each experiment
save_path = "./data/models/coordvit_mnist"
data_root = "./data/image/MNIST"
project = "CoordViT-MNIST"
description = "CoordViT on MNIST"
batch_processing_fn = coordvit_batch_processing_fn
n_segments = 16
compactness = 0.5
resize_stack_patches = (7, 7)
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
        dropout=args.dropout,
    )

# experiment arguments
extra_args = [
    ("--save-path", str, save_path, "path to save the model"),
    ("--data-root", str, data_root, "path to save the dataset"),
    
    ("--image-size",   int,   28,  "image size"),
    ("--channel-size", int,   1,   "channel size"),
    ("--patch-size",   int,   7,   "patch size"),
    ("--embed-size",   int,   512, "patch embedding size"),
    ("--num-heads",    int,   8,   "number of attention heads"),
    ("--classes",      int,   10,  "number of classes"),
    ("--num-layers",   int,   3,   "number of encoder layers"),
    ("--hidden-size",  int,   256, "encoder dimension"),
    ("--dropout",      float, 0.2, "encoder dimension"),
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
dataset.to_slic_graphs(resize_stack_patches=resize_stack_patches, n_segments=n_segments, compactness=compactness)
exp.prepare_dataset(dataset, graph_loader=False, batch_collate_fn=collate_slic_graph_patches)

# experiment run
exp.run(model_init_fn=model_init_fn, batch_processing_fn=batch_processing_fn)