from parametrized_ansatz_class import ParametrizedAnsatz
from torch import tensor
import torch

def ParametrizedAnsatz_test():
    n_qubits = 4
    toy_qnn = ParametrizedAnsatz(n_qubits)
    angle_tensor = tensor([1.9835, 1.9835, 1.9096, 1.8603], dtype=torch.float64)
    state_vector = toy_qnn.uniformAngleEmbedding(angle_tensor, 'rx')
    state_vector = torch.reshape(state_vector, shape=(2,2,2,2))
    ansatz_chromosome = [['rx', 'ry', 'ry', 'rz'],['ctrl_0', 'trgt_0', 'ctrl_1', 'trgt_1'], [None, 'ctrl_0', 'trgt_0', None], ['rz', 'pauli_y', 'rx', 'phase']]
    no_parameters = toy_qnn.calculate_number_parameters(ansatz_chromosome)
    parameters = toy_qnn.create_parameter_list(no_parameters)
    measurement = toy_qnn.simulate_circuit(state_vector, ansatz_chromosome, parameters)
    print(measurement)
    
if __name__ == '__main__':
    ParametrizedAnsatz_test()