from ansatz_simulation_class import AnsatzSimulation
import torch
import math

def main():
    n_qubits = 4
    angle_tensor = torch.tensor([1.9835, 1.9835, 0.45, 1.8603], dtype=torch.float64)
    my_ansatz = AnsatzSimulation(n_qubits)
    state_vector = my_ansatz.uniformAngleEmbedding(angle_tensor, 'ry')
    measurement = [my_ansatz.pauliZ_expectationValue(state_vector, i) for i in range(n_qubits)]
    #state_vector = torch.reshape(state_vector, shape=(2,2,2,2))
    for tensor in measurement:
        print(tensor)
    #pauli_x = my_ansatz.pauli_x_gate
    #new_state_vector = my_ansatz.simulate_one_qubit_gate(pauli_x, state_vector, 2)
    #chromossome = [['ctrl_0', 'trgt_0', 'ctrl_1', 'trgt_1'], [None, 'ctrl_0', 'trgt_0', None], ['rz_gate', 'pauli_y', 'rx_gate', 'phase']]
    #new_state_vector = my_ansatz.simulate_circuit(state_vector, chromossome)
    #print(f'new state vector:\n{new_state_vector}')
    
if __name__ == '__main__':
    main()