import torch
from torch import tensor, LongTensor
import torch.nn as nn
from ansatz_simulation_class import AnsatzSimulation

def patch_discarding(image_patches, max_patches):
    scores = tensor([torch.norm(patch) for patch in image_patches])
    selected_scores = torch.topk(scores, max_patches).indices.long()
    
    return torch.index_select(image_patches, 0, selected_scores)

class PatchExtraction(object):
    def __init__(self, patch_size: int, max_patches=80):
        self.patch_size = patch_size
        self.max_patches = max_patches
    
    def __call__(self, image):
    
        unfold = nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)
        image_patches = unfold(image)
        patch_len, n_patches = image_patches.shape
        image_patches = image_patches.view((n_patches, patch_len))
        
        return patch_discarding(image_patches, self.max_patches)
    
class Quanv_2d(object):
    def __init__(self, n_qubits, chromosome):
        self.n_qubits = n_qubits
        self.chromosome = chromosome
        self.simulator = AnsatzSimulation(n_qubits)
    
    def __call__(self, image):
        outputs = [self.simulator.simulate_circuit(patch, 'rx', self.chromosome) for patch in image]
        return tensor(outputs)