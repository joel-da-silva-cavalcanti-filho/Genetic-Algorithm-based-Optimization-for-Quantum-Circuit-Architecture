import pymoo
import torch
import random
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
#from quantum import train_model, test_model, QuantumModel, QuanvLayer

class QuantumCircuitOptimization(ElementwiseProblem):
    def __init__(self, n_qubits, n_gates, possible_gates, n_layers, patch_size, n_classes, input_size, mode, max_gates):
        super().__init__(n_var= n_qubits,
                         n_obj=1,
                         n_ieq_constr=0)
        self.n_gates = n_gates,
        self.n_qubits = n_qubits
        self.n_layers = n_layers 
        self.possible_gates = possible_gates
        self.patch_size = patch_size
        self.n_classes = n_classes
        self.input_size = input_size 
        self.mode = mode
        self.max_gates = max_gates
        
    def _evaluate(self, x, out, *args, **kwargs):

        # Amount of layers * # of qubits
        ansatz_depth = len(x) * self.n_qubits
        if ansatz_depth > self.max_gates:
            out["F"] = max(0, ansatz_depth - self.max_gates)
        else:
            # Generate quantum circuit, make the quanvolutions, flatten all the state vectors and fit it in a SVM with linear kernel to check for linear separability 
            # For the quanvolutions, make deterministic sampling of patches for the evaluation
            vector_state_training_data = np.array([0])
            tomography_labels = np.array([0])
            #quantum_layer = QuanvLayer(self.n_qubits,  self.n_layers, self.patch_size, x)
            #quantum_model = QuantumModel(quantum_layer, self.patch_size, self.n_qubits, self.mode)


            # Train my model here to check linear separability
            linear_SVM = SVC(kernel='linear')
            linear_SVM.fit(vector_state_training_data, tomography_labels)

            model_accuracy = linear_SVM.score(vector_state_training_data, tomography_labels)
            
            w = linear_SVM.coef_.flatten()
            margin = 1/np.linalg.norm(w)

        
            out["F"] = -model_accuracy

# Gotta test if sampling is working


class QuantumCircuitSampling(Sampling):
    
    quantum_gate_options = [None, 'pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard', 'ctrl', 'trgt']
    gate_options_without_cnot = [None, 'pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard']
    rotation_gate_options = ['rx_gate', 'ry_gate', 'rz_gate']
    
    def generate_layer_without_entanglement(self, n_qubits):
        layer = []
        for wire in range(n_qubits):
            random_gate = np.random.choice(self.gate_options_without_cnot)
            layer.append(random_gate)
        
        return layer
    
    def generate_layer_fully_entangled(self, n_qubits):
        for wire in range(n_qubits):
            rotation_gate = np.random.choice(self.rotation_gate_options)
    
    def generate_disjoint_cnots(self, n_qubits):
        cnot_count_layer_one = 0
        cnot_count_layer_two = 0
        layer_one = []
        layer_two = []
        for wire in range(n_qubits):
            if wire%2 == 0:
                layer_one.append(f'ctrl_{cnot_count_layer_one}')
                if wire != 0:
                    layer_two.append(f'trgt_{cnot_count_layer_two}')
                    cnot_count_layer_two+=1
            else:
                layer_one.append(f'trgt_{cnot_count_layer_one}')
                layer_two.append(f'ctrl_{cnot_count_layer_two}')
                cnot_count_layer_one += 1
            
        return layer_one, layer_two
    
    def _do(self, n_qubits, n_samples, **kwargs):

        X = []
        # p
        for individual in range(n_samples):
            print(f'Generating individual #{individual}')
            ansatz = []
            n_layers = random.randint(2, 7)
            print(f'Generating {n_layers} layers...')
            for layer in range(n_layers): 
                print(f'Generating layer #{layer}')
                gate_layer = []
                layer_type = random.randint(0,1)
                if layer_type == 0:
                    layer = self.generate_layer_without_entanglement(n_qubits)
                    ansatz.append(layer)
                elif layer_type == 1:
                    layer_one, layer_two = self.generate_disjoint_cnots(n_qubits)
                    ansatz.append(layer_one)
                    ansatz.append(layer_two) 
                       
            X.append(ansatz)

        return X
    
