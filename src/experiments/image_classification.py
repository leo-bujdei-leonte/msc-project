import argparse
import os
import pickle
from time import time

import wandb

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.training.common import device, set_seed
from src.utils.training.image_classification import train_eval_test_loop
from src.utils.misc import rename_increment
from src.utils.plotting import plot_train_val_test

_DEFAULT_ARGS = [
    ("--print-stats", int, 1, "print training statistics"),
    ("--save-model",  int, 1, "save the model"),
    ("--log-wandb",   int, 1, "log to Weights and Biases"),

    ("--num-workers", int,   0,   "number of workers for the dataloader"),
    ("--batch-size",  int,   32,  "batch size for the dataloader"),
    ("--train-split", float, 0.8, "train split for the dataset"),
    ("--val-split",   float, 0.1, "val split for the dataset"),
    ("--test-split",  float, 0.1, "test split for the dataset"),

    ("--num-exp",               int,   10,   "number of experiments to run"),
    ("--skip-count",            int,   0,    "skip first k experiments"),
    ("--num-epochs",            int,   100,  "number of epochs to train"),
    ("--lr",                    float, 5e-5, "learning rate"),
    ("--use-lr-scheduler",      int,   0,    "use ReduceLROnPlateau learning rate scheduler"),
    ("--lr-scheduler-factor",   float, 0.5,  "learning rate scheduler factor"),
    ("--lr-scheduler-patience", int,   5,    "learning rate scheduler patience"),
    ("--early-stopping",        int,   10,   "early stopping patience"),
    ("--weight-decay",          float, 0.0,  "weight decay"),
]

class Experiment():
    def __init__(self, name, description) -> None:
        self.name = name
        self.parser = argparse.ArgumentParser(description=description)
        self.dataset = None
    
    def parse_args(self, extra_args=[]):
        arg_names = set(arg[0] for arg in extra_args)
        assert "--save-path" in arg_names and "--data-root" in arg_names
        for arg in extra_args + [arg for arg in _DEFAULT_ARGS if arg[0] not in arg_names]:
            self.parser.add_argument(arg[0], type=arg[1], default=arg[2], help=arg[3])
        self.args = self.parser.parse_args()
        assert self.args.skip_count <= self.args.num_exp
            
    def prepare_dataset(self, dataset, batch_collate_fn=None):
        dataset.splits((self.args.train_split, self.args.val_split, self.args.test_split))
        dataset.loaders(
            batch_size=self.args.batch_size,
            shuffles=(True, False, False),
            collate_fn=batch_collate_fn,
        )
        self.dataset = dataset
        
    def run(self, model_init_fn):
        os.makedirs(self.args.save_path, exist_ok=True)
        for dir in ["metrics", "models", "losses", "accuracies"]:
            os.makedirs(self.args.save_path+os.sep+dir, exist_ok=True)
        
        full_metrics = []
        for i in range(self.args.skip_count, self.args.num_exp):
            model_name = str(time())
            
            print(f"Started experiment {i+1}")
            set_seed(i)
            
            model = model_init_fn(self.args).to(device)
            
            optimizer = Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            criterion = CrossEntropyLoss()
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.args.lr_scheduler_factor,
                patience=self.args.lr_scheduler_patience,
            ) if self.args.use_lr_scheduler else None
            
            if self.args.log_wandb:
                wandb.init(project=self.name, name=model_name, config=self.args)
            metrics, best_model = train_eval_test_loop(
                model,
                optimizer,
                criterion,
                self.dataset.train_loader, 
                self.dataset.val_loader, 
                self.dataset.test_loader,
                self.args.num_epochs,
                lr_scheduler=lr_scheduler,
                early_stopping=self.args.early_stopping,
                use_wandb=self.args.log_wandb,
            )
            if self.args.log_wandb:
                wandb.finish()
            full_metrics.append(metrics)
            
            pickle.dump(
                metrics,
                open(os.sep.join([self.args.save_path, "metrics", model_name+".pkl"]), "wb")
            )
            if self.args.save_model:
                torch.save(
                    best_model,
                    open(os.sep.join([self.args.save_path, "models", model_name+".pt"]), "wb")
                )
            if self.args.print_stats:
                plot_train_val_test(
                    list(range(self.args.num_epochs)), metrics["train_accs"], metrics["val_accs"], metrics["test_accs"],
                    title="Accuracy",
                    save_path=os.sep.join([self.args.save_path, "accuracies", model_name+".png"])
                )
                plot_train_val_test(
                    list(range(self.args.num_epochs)), metrics["train_losses"], metrics["val_losses"], metrics["test_losses"],
                    title="Loss",
                    save_path=os.sep.join([self.args.save_path, "losses", model_name+".png"])
                )
                
                raise NotImplementedError()
            
        pickle.dump(
            (self.args, full_metrics),
            open(os.sep.join([self.args.save_path, "full_metrics", self.name+".pkl"]), "wb")
        )

        return full_metrics
    