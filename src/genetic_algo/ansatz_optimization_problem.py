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
from quantum import train_model, test_model, QuantumModel, QuanvLayer

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
            quantum_layer = QuanvLayer(self.n_qubits,  self.n_layers, self.patch_size, x)
            quantum_model = QuantumModel(quantum_layer, self.patch_size, self.n_qubits, self.mode)


            # Train my model here to check linear separability
            linear_SVM = SVC(kernel='linear')
            linear_SVM.fit(vector_state_training_data, tomography_labels)

            model_accuracy = linear_SVM.score(vector_state_training_data, tomography_labels)
            
            w = linear_SVM.coef_.flatten()
            margin = 1/np.linalg.norm(w)

        
            out["F"] = -model_accuracy

# Gotta test if sampling is working

class QuantumCircuitSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        quantum_gate_options = [None, 'pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard', 'ctrl', 'trgt']
        gate_options_without_cnot = [None, 'pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard']
        X = []
        # p
        for individual in range(n_samples):
            tensor_network = []
            n_layers = random.randint(2, 7)
            for layer in range(n_layers): 
                gate_layer = []
                for wire in range(problem.n_qubits):
                    random_gate = np.random.choice(quantum_gate_options)
                    cnot_stack = []
                    cnot_count = -1
                    if random_gate == 'ctrl' or random_gate == 'trgt':
                        if not cnot_stack:
                            if wire == problem.n_qubits - 1:
                                cnot_wire, gate_axis = cnot_stack[0]
                                if gate_axis == 'ctrl':
                                    random_gate = f'trgt_{cnot_count}'
                                else:
                                    random_gate = f'ctrl_{cnot_count}'
                                cnot_stack.pop()
                            else:
                                cnot_stack.append([wire, random_gate])
                                cnot_count+=1
                                random_gate += f'_{cnot_count}'
                        else:
                            if random_gate == 'ctrl':
                                random_gate = f'trgt_{cnot_count}'
                            else:
                                random_gate = f'ctrl_{cnot_count}'
                            cnot_stack.pop()
                            cnot_count+=1
                     

                    if cnot_stack and wire == problem.n_qubits - 1:
                        cnot_wire, gate_axis = cnot_stack[0] 
                        gate_layer[cnot_wire] = np.random.choice(gate_options_without_cnot)

                    gate_layer.append(random_gate)
                tensor_network.append(gate_layer)    
            X.append(tensor_network)

        return np.array(X)
    
