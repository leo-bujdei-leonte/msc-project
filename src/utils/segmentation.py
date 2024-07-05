from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage import filters, graph
import torch
import numpy as np
from torch_geometric.utils.convert import from_networkx
from torchvision.transforms import Resize


def segment_image_slic_avg_pool(image, n_segments=14*14, compactness=0.5):
    assert image.shape[0] == 1
    
    image = np.array(image.squeeze(0))
    
    seg = slic(image, n_segments=n_segments, compactness=compactness, channel_axis=None)
    reg = regionprops(seg, image)
    
    intensities = torch.tensor(list(map(lambda r: r.mean_intensity, reg))).unsqueeze(-1)
    positions = torch.tensor(list(map(lambda r: r.centroid, reg)))

    tokens = torch.cat((intensities, positions), dim=-1)
    
    return tokens.float()

def extract_patches(img, seg, reg):
    imgs, masks, coords = [], [], []
    for idx in range(len(reg)):
        x_min, y_min, x_max, y_max = reg[idx].bbox
        cropped_image = img[x_min:x_max, y_min:y_max]
        cropped_mask = seg[x_min:x_max, y_min:y_max]
        cropped_mask = cropped_mask == idx+1
        imgs.append(cropped_image)
        masks.append(cropped_mask)
        coords.append(reg[idx].centroid)

    return imgs, masks, coords

def image_to_SLIC_graph(img, n_segments=14*14, compactness=0.5, save_img=False):
    assert type(img) == torch.Tensor and len(img.shape) == 3 and img.shape[0] == 1

    img = np.array(img.squeeze(0))
    seg = slic(img, n_segments=n_segments, compactness=compactness, channel_axis=None)
    reg = regionprops(seg)

    edge_boundary = filters.sobel(img)
    nx_g = graph.rag_boundary(seg, edge_boundary)

    g = from_networkx(nx_g)
    if save_img:
        g.img = img
        g.seg = seg
        g.edge_boundary = edge_boundary

    imgs, masks, coords = extract_patches(img, seg, reg)
    g.centroid = torch.Tensor([coords[label[0]-1] for label in g.labels])
    g.imgs = [imgs[label[0]-1] for label in g.labels]
    g.masks = [masks[label[0]-1] for label in g.labels]

    return g


if __name__ == "__main__":
    import pickle
    from .plotting import plot_slic_graph
    from torchvision.transforms import Resize
    
    image = pickle.load(open("./data/image/digit_example_tensor.pkl", "rb"))
    
    g = image_to_SLIC_graph(image, save_img=True)
    plot_slic_graph(g)