import pymoo
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import random
import torch.nn as nn
from pymoo.core.repair import Repair
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from genetic_hqcnn import HybridModel, train_model, validate_model
from pymoo.core.mutation import Mutation
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np


class RepairAnsatz:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        
    def _do(self, individual):
        new_size = random.randint(4, 8)
        selected_indexes = np.random.randint(0, high=len(individual), size=new_size)
    
        return [individual[index] for index in selected_indexes]

class QuantumCircuitOptimization:
    def __init__(self, n_qubits, n_gates, possible_gates, n_layers, patch_size, n_classes, input_size, mode, max_gates, dataset, batch_size, train_ratio):
        self.n_gates = n_gates,
        self.n_qubits = n_qubits
        self.n_layers = n_layers 
        self.possible_gates = possible_gates
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.input_size = input_size 
        self.mode = mode
        self.max_gates = max_gates
        
        

        train_len=int(len(dataset)*train_ratio)
        test_len=len(dataset)-int(len(dataset)*train_ratio)
        train_set= Subset(dataset,range(0,train_len))
        val_set=Subset(dataset,range(train_len,len(dataset)))
                
        
        self.train_load = DataLoader(train_set, batch_size, shuffle=True)
        self.val_load = DataLoader(val_set, batch_size, shuffle=True)
        self.repair = RepairAnsatz(n_qubits)
        
    def _evaluate(self, x, generation, individual):

        # Amount of layers * # of qubits
        ansatz_depth = len(x) * self.n_qubits
        if ansatz_depth > self.max_gates:
            x = self.repair._do(x)
            
        hqcnn = HybridModel(self.n_qubits, self.patch_size, x, self.n_classes, self.input_size, self.mode)
        optimizer = torch.optim.Adam(hqcnn.parameters(), lr=0.1, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        device = torch.device("cpu")
        n_epochs = 5
        
        filepath = f'weights/mnist_hqcnn_GA_gen_#{generation + 1}_individual_#{individual+1}'
        train_model(hqcnn, self.train_load, n_epochs, optimizer, loss_fn, filepath)
        
        val_hqcnn = HybridModel(self.n_qubits, self.patch_size, x, self.n_classes, self.input_size, self.mode)
        val_hqcnn.load_state_dict(torch.load(filepath, weights_only=True))
        avg_loss, accuracy, precision, recall, f1, auc = validate_model(hqcnn, self.val_load, loss_fn, device)
        return f1

# Gotta test if sampling is working    


class QuantumCircuitSampling:
    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gate_options_without_cnot = ['pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard']
        self.rotation_gate_options = ['rx_gate', 'ry_gate', 'rz_gate']
        self.non_parametrized_gates = ['pauli_x', 'pauli_y','pauli_z', 'phase','t','hadamard']
    
    def generate_layer_without_entanglement(self):
       return [random.choice(self.gate_options_without_cnot) for wire in range(self.n_qubits)]
    
    def generate_rotation_layer(self):
        return [random.choice(self.rotation_gate_options) for wire in range(self.n_qubits)]
    
    def place_a_single_gate(self):
        return random.choice(self.gate_options_without_cnot) 
    
    def generate_disjoint_cnots(self):
        cnot_count_layer_one = 0
        cnot_count_layer_two = 0
        layer_one = []
        layer_two = []
        
        layer_two.append(self.place_a_single_gate())
        
        for wire in range(self.n_qubits - 1):
            if wire%2 == 0:
                layer_one.append(f'ctrl_{cnot_count_layer_one}')
                layer_one.append(f'trgt_{cnot_count_layer_one}')
                cnot_count_layer_one += 1
               
            else:
                layer_two.append(f'ctrl_{cnot_count_layer_two}')
                layer_two.append(f'trgt_{cnot_count_layer_two}')
                cnot_count_layer_two += 1
            
        if self.n_qubits%2 == 0:
            layer_two.append(self.place_a_single_gate())
        else:
            layer_one.append(self.place_a_single_gate())
        
        return layer_one, layer_two
    
    def generate_non_parametrized_layer(self):
        return [random.choice(self.non_parametrized_gates) for wire in range(self.n_qubits)]

    def generate_partially_entangled_layer(self):
        pairs = [(wire,wire+1) for wire in range(self.n_qubits-1)]
        #print(pairs)
        selected_pair = random.choice(pairs)
        control, target = selected_pair
        layer = ['ctrl_0' if wire == control else
                'trgt_0' if wire == target else
                self.place_a_single_gate()
                for wire in range(self.n_qubits)]
        
        return layer
    
    def generate_individual(self):
        ansatz = []
        n_layers = random.randint(4, 10)
        print(f'Generating {n_layers} layers...')
        depth = 0
        while depth <= n_layers:
            print(f'Generating layer #{depth}')
            gate_layer = []
            layer_type = random.randint(0,4)
            if layer_type == 0:
                layer = self.generate_layer_without_entanglement()
                ansatz.append(layer)
            elif layer_type == 1:
                layer_one, layer_two = self.generate_disjoint_cnots()
                ansatz.append(layer_one)
                ansatz.append(layer_two) 
                depth+=1
            elif layer_type == 2:
                layer = self.generate_rotation_layer()
                ansatz.append(layer)
            elif layer_type == 3:
                layer = self.generate_non_parametrized_layer()
                ansatz.append(layer)
            elif layer_type == 4:
                layer = self.generate_partially_entangled_layer()
                ansatz.append(layer)
                
            depth+=1
        
        return ansatz

    def _do(self, n_samples):
        print('where are you?')
        population = []
        
        for individual in range(n_samples):
            print(f'Generating individual...')
            population.append(self.generate_individual())
                           
        return population
    
