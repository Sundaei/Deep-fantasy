import torch as tc
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import random
USE_CUDA=tc.cuda.is_available()
device=tc.device("cuda"if USE_CUDA else "cpu")
print("asdf:",device)