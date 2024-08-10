import os
import pickle
import argparse

from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from src.utils.training.common import device, set_seed
from src.utils.training.image_classification import train_eval_test_loop
from src.models.fcn import FCN
from src.utils.preprocess import random_split
from src.utils.misc import rename_increment


parser = argparse.ArgumentParser(
    description="FCN on MNIST",
)
args = [
    ("--save-path",   str, "./data/models/fcn_MNIST/", "path to save the model"),
    ("--print-stats", int, 1,                          "print training statistics"),
    ("--save-model",  int, 1,                          "save the model"),

    ("--data-root",          str,   "./data/image/MNIST", "path to save the dataset"),
    ("--num-workers",        int,   0,                    "number of workers for the dataloader"),
    ("--batch-size",         int,   32,                   "batch size for the dataloader"),
    ("--train-split",        float, 0.8,                  "train split for the dataset"),
    ("--val-split",          float, 0.1,                  "val split for the dataset"),
    ("--test-split",         float, 0.1,                  "test split for the dataset"),

    ("--num-exp",               int,   10,   "number of experiments to run"),
    ("--skip-count",            int,   0,    "skip first k experiments"),
    ("--num-epochs",            int,   100,  "number of epochs to train"),
    ("--lr",                    float, 5e-5, "learning rate"),
    ("--use-lr-scheduler",      int,   1,    "use ReduceLROnPlateau learning rate scheduler"),
    ("--lr-scheduler-factor",   float, 0.5,  "learning rate scheduler factor"),
    ("--lr-scheduler-patience", int,   5,    "learning rate scheduler patience"),
    ("--early-stopping",        int,   10,   "early stopping patience"),
    ("--weight-decay",          float, 0.0,  "weight decay"),
    
    ("--image-size",      int, 28,  "image size"),
    ("--channel-size",    int, 1,   "channel size"),
    ("--classes",         int, 10,  "number of classes"),
    ("--num-conv-layers", int, 3,   "number of convolutional layers"),
    ("--num-lin-layers",  int, 3,   "number of mlp layers"),
    ("--hidden-size",     int, 256, "hidden size"),
]
for a in args:
    parser.add_argument(a[0], type=a[1], default=a[2], help=a[3])

args = parser.parse_args()

assert args.skip_count <= args.num_exp

# data loading and preprocessing    
transform = Compose([
    Resize((args.image_size, args.image_size)),
    ToTensor(),
    Normalize(0, 1),
])
dataset = MNIST(root=args.data_root, download=True, transform=transform)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    (args.train_split, args.val_split, args.test_split),
    "MNIST",
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

full_metrics = []
os.makedirs(args.save_path, exist_ok=True)
for i in range(args.skip_count, args.num_exp):
    # experiment loop
    print(f"Started experiment {i+1}")
    
    set_seed(i)
    
    model = FCN(
        args.num_conv_layers,
        args.channel_size,
        args.hidden_size,
        args.classes,
        args.num_lin_layers,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
    ) if args.use_lr_scheduler else None

    metrics = train_eval_test_loop(
        model,
        optimizer,
        criterion,
        train_loader, 
        val_loader, 
        test_loader, 
        args.num_epochs,
        lr_scheduler=lr_scheduler,
        early_stopping=args.early_stopping,
    )
    full_metrics.append(metrics)

    pickle.dump(metrics, open(rename_increment(args.save_path+"metrics", "pkl"), "wb"))
    if args.save_model:
        raise NotImplementedError()
    if args.print_stats:
        raise NotImplementedError()

pickle.dump((args, full_metrics), open(rename_increment(args.save_path+"full_metrics", "pkl"), "wb"))