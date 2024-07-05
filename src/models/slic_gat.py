import pickle

from skimage.segmentation import slic, mark_boundaries

import matplotlib.pyplot as plt

import numpy as np



if __name__ == "__main__":
    image = pickle.load(open("./data/image/digit_example_tensor.pkl", "rb"))
    image = np.array(image.squeeze(0))
    
    seg = slic(image, n_segments=14*14, compactness=0.5, channel_axis=None)
    
    plt.imshow(mark_boundaries(image, seg, color=(255, 0, 0)))
    plt.show()