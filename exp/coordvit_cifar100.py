import os
import pickle

from torch.utils.data import random_split, DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.data.preprocess import resize_stack_slic_graph_patches, collate_slic_graph_patches
from src.models.slic_transformer import CoordViT
from src.utils.training.image_classification import train_eval_test_loop, train_epoch_coordvit, eval_coordvit
from src.utils.training.common import device, set_seed
from src.utils.data.preprocess import random_split
from src.utils.misc import rename_increment


save_path = "./data/models/coordvit_cifar100/"
        
dataset = pickle.load(open("./data/image/CIFAR100/CIFAR_SLIC_graph_32_16_15.pkl", "rb"))
dataset = resize_stack_slic_graph_patches(dataset, (8, 8)) # this is part of the model
print("Finished preprocessing")

batch_size = 32
train_dataset, val_dataset, test_dataset = random_split(dataset, [.8, .1, .1], "CIFAR100")
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_slic_graph_patches,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_slic_graph_patches,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_slic_graph_patches,
)

NUM_EXP = 10

for i in range(NUM_EXP):
    print(f"Started experiment {i+1}")
    
    set_seed(i)
    
    image_size = 32
    channel_size = 3
    patch_size = 8
    embed_size = 512
    num_heads = 8
    classes = 100
    num_layers = 3
    hidden_size = 256
    dropout = 0.2
    model = CoordViT(
        image_size,
        channel_size,
        patch_size,
        embed_size,
        num_heads,
        classes,
        num_layers,
        hidden_size,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=5e-5)
    criterion = CrossEntropyLoss()
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    num_epochs = 100
    early_stopping = 10

    metrics = train_eval_test_loop(
        model,
        optimizer,
        criterion,
        train_loader, 
        val_loader, 
        test_loader, 
        num_epochs,
        lr_scheduler=lr_scheduler,
        early_stopping=early_stopping,
        train_epoch_fn=train_epoch_coordvit,
        eval_fn=eval_coordvit,
    )

    os.makedirs(save_path, exist_ok=True)
    pickle.dump(metrics, open(rename_increment(save_path+"metrics", "pkl"), "wb"))
