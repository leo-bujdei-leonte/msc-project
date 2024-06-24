from .common import *

def train_epoch(model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                loader: DataLoader) -> tuple[float, float]:
    model.train()
    correct, total_loss, total = 0, 0, 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        
        optimizer.zero_grad()
        
        out = model(x)
        loss = criterion(out, y)
        total_loss += loss.item()
        correct += out.argmax(dim=-1).eq(y).sum().item()
        total += len(y)
        
        loss.backward()
        optimizer.step()
    
    if total == 0:
        return 0, 1
    return total_loss, correct / total

def eval(model: nn.Module, criterion: nn.Module, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    correct, total_loss, total = 0, 0, 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        
        out = model(x)
        total_loss += criterion(out, y).item()
        correct += out.argmax(dim=-1).eq(y).sum().item()
        total += len(y)
        
    if total == 0:
        return 0, 1
    return total_loss, correct / total

def train_test_loop(model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                    train_loader: DataLoader, test_loader: DataLoader,
                    num_epochs: int, lr_scheduler: LRScheduler = None
                    ) -> tuple[list[float]]:
    train_accs, test_accs, train_losses, test_losses = [], [], [], []
    for i in range(1, num_epochs+1):
        interval = time()
        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader)
        test_loss, test_acc = eval(model, criterion, test_loader)
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=i, metrics=train_loss)
        interval = time() - interval
        
        print(f"Epoch {i:03d}: train loss {train_loss:.4f}, train accuracy {train_acc:.3f}, test accuracy {test_acc:.3f} (total time {int(interval)}s)")
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    
    return train_losses, train_accs, test_losses, test_accs