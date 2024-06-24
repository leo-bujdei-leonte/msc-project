import os
import pickle

from time import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
