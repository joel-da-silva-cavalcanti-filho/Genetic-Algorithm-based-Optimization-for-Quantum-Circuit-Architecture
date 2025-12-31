import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch import tensor

class SampledDataset4Training(Dataset):
    def __init__(self, dataset, transform=None):
        images, labels = zip(*dataset)
        labels = [label%2 for label in labels]
        self.labels = tensor(labels)
        self.images = torch.stack(images)
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        
        if self.transform:
            sample = self.transform(self.images[index])
        else:
            sample = self.images[index]
        
        
        return sample, self.labels[index]