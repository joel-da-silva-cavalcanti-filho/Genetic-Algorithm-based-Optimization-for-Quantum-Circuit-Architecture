import torch
from torch import tensor
import torch.nn as nn

class PatchExtraction(object):
    def __init__(self, patch_size: int):
        self.patch_size = patch_size
    
    def __call__(self, image):
    
        unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)
        image_patches = unfold(image)
        patch_len, n_patches = image_patches.shape
        image_patches = image_patches.view((n_patches, patch_len))
        
        return image_patches
    