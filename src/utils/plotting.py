import os
from typing import Iterable

import matplotlib.pyplot as plt
from skimage import graph, color

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
    
    plt.close()

def plot_train_val_test(x: Iterable, y_train: Iterable, y_val: Iterable, y_test: Iterable,
                        title: str, save_path: str, figsize: tuple[int, int] = (12, 8)) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.clf()

    plt.figure(figsize=figsize)
    plt.plot(x, y_train)
    plt.plot(x, y_val)
    plt.plot(x, y_test)
    plt.legend(["Train", "Val", "Test"])
    plt.title(title)
    
    if os.path.isfile(save_path):
        os.remove(save_path)
    plt.savefig(save_path)
    
    plt.close()

def plot_slic_graph(g, figsize=(10, 8), edge_cmap='viridis', edge_width=1.2, fraction=0.05):
    plt.clf()
    
    plt.figure(figsize=figsize)
    res = graph.show_rag(
        g.seg,
        graph.rag_boundary(g.seg, g.edge_boundary),
        color.gray2rgb(g.img),
        edge_cmap=edge_cmap,
        edge_width=edge_width,
    )
    plt.colorbar(res, fraction=fraction)
    
    plt.show()
