import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class SampledDataset4Training(Dataset):
    def __init__(self, dataset):
        images, labels = zip(*dataset)
        self.images = torch.stack(images).squeeze(0)
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        sample = {'image': self.images[index], 'label': self.labels[index]}
        
        return sample