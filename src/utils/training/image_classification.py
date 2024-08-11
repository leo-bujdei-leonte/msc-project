import os
import pickle
from time import time
from copy import deepcopy

import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .common import *
from ..plotting import plot_train_test


def default_batch_processing_fn(batch, model):
    x, y = batch[0].to(device), batch[1].to(device)
    out = model(x)
    
    return out, y

def gcn_batch_processing_fn(batch, model):
    out = model(
        batch.x.to(device),
        batch.edge_index.to(device),
        batch.batch.to(device)
    )
    
    return out, batch.y.to(device)


def train_epoch_gcn(model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                loader: DataLoader, normalise_loss=True) -> tuple[float, float]:
    model.train()
    correct, total_loss, total = 0, 0, 0
    for batch in loader:
        x, edge_index, batch, y = batch.x.to(device), batch.edge_index.to(device), batch.batch.to(device), batch.y.to(device)
        
        optimizer.zero_grad()
        
        out = model(x, edge_index, batch)
        loss = criterion(out, y)
        total_loss += loss.item()
        correct += out.argmax(dim=-1).eq(y).sum().item()
        total += len(y)
        
        loss.backward()
        optimizer.step()
    
    if total == 0:
        return 0, 1
    
    if normalise_loss:
        total_loss /= total
    
    return total_loss, correct / total
    

def train_epoch(model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                loader: DataLoader, batch_processing_fn=default_batch_processing_fn,
                normalise_loss=True) -> tuple[float, float]:
    model.train()
    correct, total_loss, total = 0, 0, 0
    for batch in loader:
        optimizer.zero_grad()
        
        out, y = batch_processing_fn(batch, model)
        loss = criterion(out, y)
        total_loss += loss.item()
        correct += out.argmax(dim=-1).eq(y).sum().item()
        total += len(y)
        
        loss.backward()
        optimizer.step()
    
    if total == 0:
        return 0, 1
    
    if normalise_loss:
        total_loss /= total
    
    return total_loss, correct / total

def train_epoch_coordvit(model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                loader: DataLoader, normalise_loss=True) -> tuple[float, float]:
    model.train()
    correct, total_loss, total = 0, 0, 0
    for batch in loader:
        x, coords, mask, y = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        
        optimizer.zero_grad()
        
        out = model(x, coords, mask)
        loss = criterion(out, y)
        total_loss += loss.item()
        correct += out.argmax(dim=-1).eq(y).sum().item()
        total += len(y)
        
        loss.backward()
        optimizer.step()
    
    if total == 0:
        return 0, 1
    
    if normalise_loss:
        total_loss /= total
    
    return total_loss, correct / total

def eval(model: nn.Module, criterion: nn.Module, loader: DataLoader,
         batch_processing_fn=default_batch_processing_fn,
         normalise_loss=True) -> tuple[float, float]:
    model.eval()
    correct, total_loss, total = 0, 0, 0
    for batch in loader:
        out, y = batch_processing_fn(batch, model)
        total_loss += criterion(out, y).item()
        correct += out.argmax(dim=-1).eq(y).sum().item()
        total += len(y)
        
    if total == 0:
        return 0, 1
    if normalise_loss:
        total_loss /= total
    return total_loss, correct / total

def eval_coordvit(model: nn.Module, criterion: nn.Module, loader: DataLoader, normalise_loss=True) -> tuple[float, float]:
    model.eval()
    correct, total_loss, total = 0, 0, 0
    for batch in loader:
        x, coords, mask, y = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
        
        out = model(x, coords, mask)
        total_loss += criterion(out, y).item()
        correct += out.argmax(dim=-1).eq(y).sum().item()
        total += len(y)
        
    if total == 0:
        return 0, 1
    if normalise_loss:
        total_loss /= total
    return total_loss, correct / total

def train_test_loop(model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                    train_loader: DataLoader, test_loader: DataLoader,
                    num_epochs: int, lr_scheduler: LRScheduler = None,
                    save_path: str = None, plot=False,
                    batch_processing_fn=default_batch_processing_fn,
                    normalise_loss=True,
                    ) -> tuple[list[float]]:
    assert save_path is not None or plot == False
    
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        if os.path.exists(save_path + "model.pt"):
            model.load_state_dict(torch.load(save_path + "model.pt"))
        if os.path.exists(save_path + "metrics.pkl"):
            metrics = pickle.load(open(save_path + "metrics.pkl", "rb"))
            train_accs, test_accs, train_losses, test_losses = metrics
            
    
    for i in range(1+len(train_accs), num_epochs+1):
        interval = time()

        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader, normalise_loss=normalise_loss, batch_processing_fn=batch_processing_fn)
        test_loss, test_acc = eval(model, criterion, test_loader, normalise_loss=normalise_loss, batch_processing_fn=batch_processing_fn)
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=i, metrics=train_loss)

        interval = time() - interval
        
        print(
            f"Epoch {i:03d}: train loss {train_loss:.5f},",
            f"train accuracy {train_acc:.5f},",
            f"test accuracy {test_acc:.5f}, "
            f"time {int(interval)}s"
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc) # worse because pre-training per batch
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if save_path is not None:
            torch.save(model.state_dict(), save_path + "model.pt")
            pickle.dump(
                (train_accs, test_accs, train_losses, test_losses),
                open(save_path + "metrics.pkl", "wb"),
            )
        
        if plot:
            plot_train_test(list(range(1, i+1)), train_losses, test_losses,
                            "Loss", save_path+"loss.png")
            plot_train_test(list(range(1, i+1)), train_accs, test_accs,
                            "Classification accuracy", save_path+"accuracy.png")

    return train_losses, train_accs, test_losses, test_accs

def train_eval_test_loop(model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                    train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                    num_epochs: int, lr_scheduler: LRScheduler = None,
                    batch_processing_fn=default_batch_processing_fn, early_stopping=float("inf"),
                    normalise_loss=True, use_wandb=False,
                    ) -> tuple[list[float]]:    
    train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = [], [], [], [], [], []
            
    best_model = deepcopy(model)
    max_val_acc = float("-inf")
    num_not_max = 0
    for i in range(1+len(train_accs), num_epochs+1):
        if num_not_max >= early_stopping:
            print("Early stopping")
            break
        
        interval = time()

        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader, normalise_loss=normalise_loss, batch_processing_fn=batch_processing_fn)
        val_loss, val_acc = eval(model, criterion, val_loader, normalise_loss=normalise_loss, batch_processing_fn=batch_processing_fn)
        test_loss, test_acc = eval(model, criterion, test_loader, normalise_loss=normalise_loss, batch_processing_fn=batch_processing_fn)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            best_model = deepcopy(model)
            num_not_max = 0
            new_flag = True
        else:
            num_not_max += 1
            new_flag = False
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=i, metrics=train_loss)

        interval = time() - interval
        
        print(
            f"Epoch {i:03d}: train loss {train_loss:.5f},",
            f"train accuracy {train_acc:.5f},",
            f"val loss {val_loss:.5f},",
            f"val accuracy {val_acc:.5f},",
            f"test loss {test_loss:.5f},",
            f"test accuracy {test_acc:.5f},",
            f"time {int(interval)}s",
            "NEW" if new_flag else "",
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc) # worse because pre-training per batch
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        if use_wandb:
            wandb.log({
                "epoch": i,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc
            })

    return {
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_accs": test_accs,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_losses": test_losses,
    }, best_model