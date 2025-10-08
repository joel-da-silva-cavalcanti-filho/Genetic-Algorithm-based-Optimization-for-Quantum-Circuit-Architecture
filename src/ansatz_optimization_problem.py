import pymoo
import torch
import random
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
from quantum import train_model, test_model, QuantumModel, QuanvLayer

class QuantumCircuitOptimization(ElementwiseProblem):
    def __init__(self, n_qubits, n_gates, possible_gates, n_layers, patch_size, n_classes, input_size, mode):
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
        
    def _evaluate(self, x, out, *args, **kwargs):
        quantum_layer = QuanvLayer(self.n_qubits,  self.n_layers, self.patch_size, x)
        quantum_model = QuantumModel(quantum_layer, self.patch_size, self.n_qubits, self.mode)
        # Train my model here
        train_model(quantum_model, )
        f1 = test_model()
        out["F"] = -f1

class QuantumCircuitSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        gate_operation = ['pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard']
        X = []
        # p
        for individual in range(n_samples):
            tensor_network = []
            for wire in range(problem.n_qubits): 
                qubit = []
                no_gates = random.randint(2, 7)
                for gate in range(no_gates):
                    qubit.append(np.random.choice(gate_operation))
                tensor_network.append(qubit)    
            X.append(tensor_network)

        return np.array(X)