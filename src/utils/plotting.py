from typing import Iterable
import matplotlib.pyplot as plt
import os

def plot_train_test(x: Iterable, y_train: Iterable, y_test: Iterable,
                    title: str, save_path: str, figsize: tuple[int, int] = (12, 8)) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.clf()

    plt.figure(figsize=figsize)
    plt.plot(x, y_train)
    plt.plot(x, y_test)
    plt.legend(["Train", "Test"])
    plt.title(title)
    
    if os.path.isfile(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    
    plt.clf()