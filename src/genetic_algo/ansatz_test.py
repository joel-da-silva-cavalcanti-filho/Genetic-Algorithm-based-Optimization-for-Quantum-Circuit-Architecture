from ansatz_simulation_class import AnsatzSimulation
import torch

def main():
    angle_tensor = torch.tensor([1.9835, 1.9835, 1.9096, 1.8603], dtype=torch.float64)
    my_ansatz = AnsatzSimulation(n_qubits=4)
    state_vector = my_ansatz.uniformAngleEmbedding(angle_tensor, 'rx')
    state_vector = torch.reshape(state_vector, shape=(2,2,2,2))
    #pauli_x = my_ansatz.pauli_x_gate
    #new_state_vector = my_ansatz.simulate_one_qubit_gate(pauli_x, state_vector, 2)
    chromossome = [['ctrl_0', 'trgt_0', 'ctrl_1', 'trgt_1'], ['ctrl_0', 'trgt_0', 'ctrl_1'], ['rz_gate', 'pauli_y', 'rx_gate', 'phase']]
    new_state_vector = my_ansatz.simulate_circuit(state_vector, chromossome)
    print(f'new state vector:\n{new_state_vector}')
    
if __name__ == '__main__':
    main()