from .common import *
from ..plotting import plot_train_test

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
    return total_loss, correct / total # NOTE loss not normalised

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
    return total_loss, correct / total # NOTE loss not normalised

def train_test_loop(model: nn.Module, optimizer: Optimizer, criterion: nn.Module,
                    train_loader: DataLoader, test_loader: DataLoader,
                    num_epochs: int, lr_scheduler: LRScheduler = None,
                    save_path: str = None, plot=False
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

        train_loss, train_acc = train_epoch(model, optimizer, criterion, train_loader)
        test_loss, test_acc = eval(model, criterion, test_loader)
        if lr_scheduler is not None:
            lr_scheduler.step(epoch=i, metrics=train_loss)

        interval = time() - interval
        
        print(
            f"Epoch {i:03d}: train loss {train_loss:.4f},",
            f"train accuracy {train_acc:.3f},",
            f"test accuracy {test_acc:.3f}, "
            f"time {int(interval)}s"
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
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