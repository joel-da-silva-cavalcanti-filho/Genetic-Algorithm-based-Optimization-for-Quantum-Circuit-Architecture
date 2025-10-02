import pymoo
import torch
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
import numpy as np
from quantum import train_model, test_model, QuantumModel, QuanvLayer

class QuantumCircuitOptimization(ElementwiseProblem):
    def __init__(self, n_qubits, n_gates, possible_gates):
        super().__init__(n_var= n_qubits,
                         n_obj=6,
                         n_ieq_constr=0)
        self.n_gates = n_gates,
        self.n_qubits = n_qubits
        self.possible_gates = possible_gates

        
    def _evaluate(self, x, out, *args, **kwargs):

        quantum_layer = QuanvLayer(self.n_qubits,)
        quantum_model = QuantumModel(self.n_qubits)
        # Train my model here
        train_model()
        avg_loss, accuracy, precision, recall, f1, auc = test_model()
        out["F"] = np.column_stack([avg_loss, accuracy, precision, recall, f1, auc])

class QuantumCircuitSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        gate_operation = ['pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard']
        X = np.full((n_samples, 1))

        # p
        for individual in range(n_samples):
            tensor_network = []
            for wire in range(problem.n_qubits): 
                X[individual, 0] = ", ".join([np.random.choice(problem.possible_gates) for _ in range(problem.n_gates)])