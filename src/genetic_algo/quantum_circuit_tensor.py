import torch
import math
import cmath

class AnsatzSimulation():

    H = [[(1/math.sqrt(2)), (1/math.sqrt(2))], [(1/math.sqrt(2)), -(1/math.sqrt(2))]]
    theta_value = 2*math.pi

    hadamard_gate = torch.tensor(H, dtype=torch.complex64)

    rx_gate = torch.tensor([[math.cos(theta_value/2), -1j*math.sin(theta_value/2)],
                            [-1j*math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)

    ry_gate = torch.tensor([[math.cos(theta_value/2), -math.sin(theta_value/2)], [math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)

    rz_gate = torch.tensor([[-cmath.exp(-1.0j*theta_value/2), 0.0], [0.0, cmath.exp(-1.0j*theta_value/2)]], dtype=torch.complex64)
    
    pauli_x_gate = torch.tensor([[0.0, 1.0],[1.0 ,0.0]], dtype=torch.complex64)

    pauli_y_gate = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]],dtype=torch.complex64)
    pauli_z_gate = torch.tensor([[1.0, 0.0], [0.0,-1.0]], dtype=torch.complex64)

    phase_gate = torch.tensor([[1.0, 0.0], [0.0, 1.0j]], dtype=torch.complex64)
    t_gate = torch.tensor([[1.0, 0.0], [0.0, cmath.exp(math.pi*1.0j/4)]], dtype=torch.complex64)
    eye_tensor = torch.eye(2, dtype=torch.complex64)

    gate_operation = {
        'pauli_x': pauli_x_gate,
        'pauli_y': pauli_y_gate,
        'pauli_z': pauli_z_gate,
        'rx_gate': rx_gate,
        'ry_gate': ry_gate,
        'rz_gate': rz_gate,
        'phase': phase_gate,
        't': t_gate,
        'hadamard': hadamard_gate
    }

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    # Could do the embedding changing the rotation gates for each angle

    def rx_rotation_states(self, patch_vector):
        qubit_count = 0
        for theta_value in patch_vector:
            if qubit_count == 0:
                initial_state_vector = torch.tensor([[math.cos(theta_value/2), -1j*math.sin(theta_value/2)],
                        [-1j*math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)
            else:
                psi = torch.tensor([[math.cos(theta_value/2), -1j*math.sin(theta_value/2)],
                        [-1j*math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)
                initial_state_vector = torch.kron(initial_state_vector, psi)

            qubit_count += 1

        return initial_state_vector
       

    def ry_rotation_states(self, patch_vector):
        qubit_count = 0
        for theta_value in patch_vector:
            if qubit_count == 0:
                initial_state_vector = torch.tensor([[math.cos(theta_value/2), -math.sin(theta_value/2)], [math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)
            else:
                psi = torch.tensor([[math.cos(theta_value/2), -math.sin(theta_value/2)], [math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)
                initial_state_vector = torch.kron(initial_state_vector, psi)

            qubit_count += 1

        return initial_state_vector
    
    def rz_rotation_states(self, patch_vector):
        qubit_count = 0
        for theta_value in patch_vector:
            if qubit_count == 0:
                initial_state_vector = torch.tensor([[-cmath.exp(-1.0j*theta_value/2), 0.0], [0.0, cmath.exp(-1.0j*theta_value/2)]], dtype=torch.complex64)

            else:
                psi = torch.tensor([[-cmath.exp(-1.0j*theta_value/2), 0.0], [0.0, cmath.exp(-1.0j*theta_value/2)]], dtype=torch.complex64)
                initial_state_vector = torch.kron(initial_state_vector, psi)

            qubit_count += 1

        return initial_state_vector

    def angle_embedding(self, patch_vector, rotation_gate):
        eye_tensor = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        eye_tensor[0] = 1.0
        if rotation_gate == 'rx':
            return self.rx_rotation_states(patch_vector) @ eye_tensor

        elif rotation_gate == 'ry':
            return self.ry_rotation_states(patch_vector) @ eye_tensor

        elif rotation_gate == 'rz':
            return self.rz_rotation_states(patch_vector)  @ eye_tensor
        

    def simulate_one_qubit_gate(self, selected_gate, state_vector, selected_qubit):
            start_position = 1
            if selected_qubit == 0:
                tensor_gate = selected_gate
            else:
                tensor_gate = self.eye_tensor

            for wire in range(start_position, self.n_qubits):
                if wire != selected_qubit:
                    tensor_gate = torch.kron(self.eye_tensor, tensor_gate)
                else:
                    tensor_gate = torch.kron(tensor_gate, selected_gate)
            
            return torch.matmul(tensor_gate, state_vector)


    def simulate_circuit(self, patch_vector: torch.tensor, ansatz_chromosome: list):
        unitary_vectors = self.angle_embedding(patch_vector, 'ry')
        for layer in ansatz_chromosome:
            qubit_count = 0
            for gate in layer:
                if gate in self.gate_operation:
                    unitary_vectors = self.simulate_one_qubit_gate(self.gate_operation[gate], unitary_vectors, qubit_count)
                elif gate is not None:
                    if gate[0:4] == 'ctrl':
                        gate_index = gate[5]
                        
                qubit_count+=1

