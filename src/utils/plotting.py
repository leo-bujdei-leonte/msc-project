from typing import Iterable
import matplotlib.pyplot as plt

def plot(x: Iterable, y: Iterable, path: str, title: str = None, figsize: tuple[int, int] = (12, 8)):
    plt.figure(figsize=figsize)
    plt.plot(x, y)
    if title is not None:
        plt.title(title)
    plt.savefig(path)