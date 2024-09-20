import os
import pickle

import torch
from torchvision.transforms import Resize
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data import Data

import numpy as np


def random_split(data, ratios, dataset_name):
    if len(ratios) == 2:
        train_ratio, test_ratio = ratios
        val_ratio = 0
    elif len(ratios) == 3:
        train_ratio, val_ratio, test_ratio = ratios
    else:
        raise ValueError("ratios must be of length 2 or 3")
    
    save_path = os.sep.join([".", "data", "split", dataset_name, f"{train_ratio}-{val_ratio}-{test_ratio}.pkl"])
    if os.path.isfile(save_path):
        idx_train, idx_val, idx_test = pickle.load(open(save_path, "rb"))
        print("Loaded existing data split")
        
    else:
        n = len(data)
        t = [int(train_ratio*n), int((train_ratio+val_ratio)*n)]
        
        idx = torch.randperm(n)
        idx_train, idx_val, idx_test = idx[:t[0]], idx[t[0]:t[1]], idx[t[1]:]
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pickle.dump((idx_train, idx_val, idx_test), open(save_path, "wb"))
        print("Created new data split")
    
    train_data, val_data, test_data = [], [], []
    for i in idx_train:
        train_data.append(data[i])
    for i in idx_val:
        val_data.append(data[i])
    for i in idx_test:
        test_data.append(data[i])
    
    return train_data, val_data, test_data

def resize_stack_slic_graph_patches(data, size):
    r = Resize(size)

    for idx, g in enumerate(data):
        for i in range(len(g.imgs)):
            g.imgs[i] = (torch.Tensor(g.imgs[i] * g.masks[i]) / torch.sum(torch.Tensor(g.masks[i]), dim=(0, 1))).unsqueeze(0)
        # g.imgs = [torch.Tensor(g.imgs[i] * g.masks[i]).unsqueeze(0) for i in range(len(g.imgs))] 
        g.imgs = [r(img) for img in g.imgs]
        g.imgs = torch.cat(g.imgs, dim=0)

        if len(g.imgs.shape) == 3:
            g.imgs = g.imgs.unsqueeze(1)
        
        if idx % 1000 == 0:
            print("Processed graph", idx)

    return data

def collate_slic_graph_patches(batch):
    lengths = torch.tensor([len(g.imgs) for g in batch])
    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    imgs = pad_sequence([g.imgs for g in batch], batch_first=True)
    coords = pad_sequence([g.centroid for g in batch], batch_first=True)

    y = torch.tensor([g.y for g in batch], dtype=torch.long)

    return imgs, coords, mask, y

def lrgb_statistics(img, mask):
    if len(img.shape) == 3:
        expanded_mask = mask.reshape(1, mask.shape[0], mask.shape[1]).repeat(img.shape[0], axis=0)
        masked_img = np.ma.masked_array(img, ~expanded_mask)
        axes = (-1, -2)
        res = torch.tensor(np.concatenate([
            masked_img.mean(axis=axes),
            masked_img.std(axis=axes),
            masked_img.min(axis=axes),
            masked_img.max(axis=axes),
        ]))
        return res
    
    elif len(img.shape) == 2:
        masked_img = np.ma.masked_array(img, ~mask)
        res = torch.tensor(np.array([
            masked_img.mean(),
            masked_img.std(),
            masked_img.min(),
            masked_img.max(),
        ]))
        return res
    
    else:
        raise NotImplementedError()

def slic_graph_patches_to_lrgb_stats(data):
    # each patch becomes a 4c-dimensional embedding with mean, std, min, max
    for idx, g in enumerate(data):
        g.x = []
        for i in range(len(g.imgs)):
            g.x.append(lrgb_statistics(g.imgs[i], g.masks[i]))
        g.x = torch.stack(g.x, dim=0)
        
        g.imgs, g.masks = None, None
        
        if idx % 1000 == 0:
            print("Processed graph", idx)
    
    return data

def generate_all_edges(h, w):
    directions = [
        (-1, 0),  # left
        (-1, -1), # top-left
        (0, -1),  # top
        (1, -1),  # top-right
        (1, 0),   # right
        (1, 1),   # bottom-right
        (0, 1),   # bottom
        (-1, 1)   # bottom-left
    ]
    
    # Generate all pixel coordinates (x, y)
    x_coords = torch.arange(w)
    y_coords = torch.arange(h)
    
    # Create a grid of all pixel coordinates
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    
    # Flatten the grid to get all (x, y) pairs
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    
    # Initialize lists to collect valid edges
    source_indices = []
    target_indices = []
    
    # Loop through all directions
    for dx, dy in directions:
        # Compute the target coordinates (x + dx, y + dy)
        target_x = grid_x + dx
        target_y = grid_y + dy
        
        # Determine valid edges (those within the image bounds)
        valid_mask = (target_x >= 0) & (target_x < w) & (target_y >= 0) & (target_y < h)
        
        # Extract valid coordinates
        valid_source_indices = grid_y[valid_mask] * w + grid_x[valid_mask]
        valid_target_indices = target_y[valid_mask] * w + target_x[valid_mask]
        
        # Append to the lists
        source_indices.append(valid_source_indices)
        target_indices.append(valid_target_indices)
    
    # Concatenate all valid edges
    all_source_indices = torch.cat(source_indices)
    all_target_indices = torch.cat(target_indices)
    
    # Stack the source and target indices to form the edges
    edges = torch.stack([all_source_indices, all_target_indices], dim=0)
    
    return edges

def image_to_pygraph(data):
    img, y = data
    c, h, w = img.shape
    
    edge_index = generate_all_edges(h, w)
    
    x = img.reshape(c, -1).T
    
    coords = torch.cartesian_prod(torch.arange(h), torch.arange(w))
    
    return Data(x=x, edge_index=edge_index, y=y, coords=coords)

if __name__ == "__main__":
    import pickle
    
    image = pickle.load(open("./data/image/example.pkl", "rb"))
    print(image_to_pygraph((image, 1)).coords.shape)
