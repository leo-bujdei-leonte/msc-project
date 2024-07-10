import os
import pickle

import torch
from torchvision.transforms import Resize
from torch.nn.utils.rnn import pad_sequence

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

    for g in data:
        for i in range(len(g.imgs)):
            g.imgs[i] = (torch.Tensor(g.imgs[i] * g.masks[i]) / torch.sum(torch.Tensor(g.masks[i]), dim=(0, 1))).unsqueeze(0)
        # g.imgs = [torch.Tensor(g.imgs[i] * g.masks[i]).unsqueeze(0) for i in range(len(g.imgs))] 
        g.imgs = [r(img) for img in g.imgs]
        g.imgs = torch.cat(g.imgs, dim=0)

        if len(g.imgs.shape) == 3:
            g.imgs = g.imgs.unsqueeze(1)

    return data

def collate_slic_graph_patches(batch):
    lengths = torch.tensor([len(g.imgs) for g in batch])
    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)

    imgs = pad_sequence([g.imgs for g in batch], batch_first=True)
    coords = pad_sequence([g.centroid for g in batch], batch_first=True)

    y = torch.tensor([g.y for g in batch], dtype=torch.long)

    return imgs, coords, mask, y
