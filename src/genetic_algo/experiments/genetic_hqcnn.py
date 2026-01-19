
from collections import Counter
import itertools
from ansatz_simulation_class import AnsatzSimulation
import torch
import math
import torch.nn as nn
import time
from torch import tensor
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class QuanvLayer(nn.Module):
    def __init__(self, n_qubits, patch_size, chromosome, mode="frozen"):
        super().__init__()
        self.n_qubits = n_qubits
        self.patch_size = patch_size
        self.chromosome = chromosome
        
        gates = list(itertools.chain.from_iterable(chromosome))
        gate_count = Counter(gates)
        n_params = gate_count['rx_gate'] + gate_count['ry_gate'] + gate_count['rz_gate']
        if mode == "frozen":
            self.ansatz_params = torch.rand(n_params)*math.pi
        elif mode == "trainable":
            self.ansatz_params = nn.Parameter(torch.rand(n_params)*math.pi)
            
        self.qlayer = AnsatzSimulation(n_qubits)
        
    def multichannel_quanv_2d(self, x):
        
        outputs = [[[self.qlayer.simulate_circuit(patch, 'ry', self.chromosome, self.ansatz_params, True) for patch in patches] for patches in channel] for channel in x]
        n_images, n_channels, n_patches, patch_size = tensor(outputs).shape
        image_size = int((n_patches * patch_size) ** 0.5)
        
        return tensor(outputs).view(n_images, image_size, image_size, n_channels)
    
    def greyscale_quanv_2d(self, x):
        
        outputs = [[self.qlayer.simulate_circuit(patch, 'rx', self.chromosome, self.ansatz_params, True) for patch in patches] for patches in x]
        
        return tensor(outputs)

    def forward(self, x):
        if len(x.shape) == 4:
            return self.multichannel_quanv_2d(x)
        elif len(x.shape) == 3:
            return self.greyscale_quanv_2d(x)
        
    
 
class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)

class HybridModel(nn.Module):
    def __init__(self, n_qubits, patch_size, chromosome, num_classes, input_size, mode='frozen'):
        super().__init__()
        self.quanv_layer = QuanvLayer(
            n_qubits=n_qubits,
            patch_size=patch_size,
            chromosome=chromosome,
            mode=mode
        )
        feature_size = input_size**2
        self.fc1 = nn.Linear(feature_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.norm = nn.LayerNorm(feature_size)
        #self.conv1 = nn.Conv2d(in_channels = 1, output_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
        self.fc = nn.Linear(feature_size, num_classes)
        self.l2norm = L2NormalizationLayer(dim=1)
        
    def forward(self, x):
        print("Passing through quanvolution layer...")
        start_time = time.perf_counter()
        x = self.quanv_layer.forward(x)
        end_time = time.perf_counter()
        print(f'Quanvolution processing time: {end_time - start_time}')  
        x = x.flatten(start_dim=1)
        #print(x.shape)
        x = self.l2norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
    

def train_model(model, train_loader, num_epochs, optimizer, loss_fn, filepath):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        epoch_loss = 0.0
        total = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (inputs, labels) in progress_bar:
        
            optimizer.zero_grad()  # Zero out previous gradients
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)  # Calculate loss
            
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loss.backward()  # Backpropagate to calculate gradients
            optimizer.step() # Update weights
            
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/(i+1):.4f}, Acc: {correct / total:.4f}")
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}")
            # Print every 10 batches
            
    torch.save(model.state_dict(), filepath)  
    
def validate_model(model, dataloader, criterion, device):
    model.eval()    
    model.to(device)

    total_loss = 0.0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
#            images = images.to(device)
#            labels = labels.squeeze().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy() if outputs.shape[1] > 1 else torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

    return avg_loss, accuracy, precision, recall, f1, auc

  