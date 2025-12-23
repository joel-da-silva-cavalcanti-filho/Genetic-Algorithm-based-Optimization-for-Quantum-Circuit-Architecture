import torch
from torch import tensor

class CreatePatches(object):
    def __init__(self, patch_size: int):
        self.patch_size = patch_size
    
    def __call__(self, sample):
        image = sample['image']
        batch, channel, height, width = image.shape
    
        patches = [image[:,:,jump:jump+4,jump:jump+4] for jump in range(0, height, self.patch_size)]
        
        