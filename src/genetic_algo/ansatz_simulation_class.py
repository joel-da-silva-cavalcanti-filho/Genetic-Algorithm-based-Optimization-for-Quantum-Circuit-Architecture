import torch
import math
import cmath
from torch import tensor, tensordot, cat
import random

class AnsatzSimulation():

    H = [[(1/math.sqrt(2)), (1/math.sqrt(2))], [(1/math.sqrt(2)), -(1/math.sqrt(2))]]
    theta = math.pi/2

    hadamard_gate = tensor(H, dtype=torch.complex64)

    rx_gate = tensor([[math.cos(theta/2), -1j*math.sin(theta/2)],
                            [-1j*math.sin(theta/2), math.cos(theta/2)]], dtype=torch.complex64)

    ry_gate = tensor([[math.cos(theta/2), -math.sin(theta/2)], [math.sin(theta/2), math.cos(theta/2)]], dtype=torch.complex64)

    rz_gate = tensor([[-cmath.exp(-1.0j*theta/2), 0.0], [0.0, cmath.exp(-1.0j*theta/2)]], dtype=torch.complex64)
    
    pauli_x_gate = tensor([[0.0, 1.0],[1.0 ,0.0]], dtype=torch.complex64)

    pauli_y_gate = tensor([[0.0, -1.0j], [1.0j, 0.0]],dtype=torch.complex64)
    pauli_z_gate = tensor([[1.0, 0.0], [0.0,-1.0]], dtype=torch.complex64)

    phase_gate = tensor([[1.0, 0.0], [0.0, 1.0j]], dtype=torch.complex64)
    t_gate = tensor([[1.0, 0.0], [0.0, cmath.exp(math.pi*1.0j/4)]], dtype=torch.complex64)
    cnot_gate = tensor([[[[1.+0.j, 0.+0.j],
          [0.+0.j, 0.+0.j]],

         [[0.+0.j, 1.+0.j],
          [0.+0.j, 0.+0.j]]],


        [[[0.+0.j, 0.+0.j],
          [0.+0.j, 1.+0.j]],

         [[0.+0.j, 0.+0.j],
          [1.+0.j, 0.+0.j]]]], dtype=torch.complex64)
    
   
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
    
    non_parametrized_gates = {
        'pauli_x': pauli_x_gate,
        'pauli_y': pauli_y_gate,
        'pauli_z': pauli_z_gate,
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
                initial_state_vector = tensor([[math.cos(theta_value/2), -1j*math.sin(theta_value/2)],
                        [-1j*math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)
            else:
                psi = tensor([[math.cos(theta_value/2), -1j*math.sin(theta_value/2)],
                        [-1j*math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)
                initial_state_vector = torch.kron(initial_state_vector, psi)

            qubit_count += 1

        return initial_state_vector
       

    def ry_rotation_states(self, patch_vector):
        qubit_count = 0
        for theta_value in patch_vector:
            if qubit_count == 0:
                initial_state_vector = tensor([[math.cos(theta_value/2), -math.sin(theta_value/2)], [math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)
            else:
                psi = tensor([[math.cos(theta_value/2), -math.sin(theta_value/2)], [math.sin(theta_value/2), math.cos(theta_value/2)]], dtype=torch.complex64)
                initial_state_vector = torch.kron(initial_state_vector, psi)

            qubit_count += 1

        return initial_state_vector
    
    def rz_rotation_states(self, patch_vector):
        qubit_count = 0
        for theta_value in patch_vector:
            if qubit_count == 0:
                initial_state_vector = tensor([[-cmath.exp(-1.0j*theta_value/2), 0.0], [0.0, cmath.exp(-1.0j*theta_value/2)]], dtype=torch.complex64)

            else:
                psi = tensor([[-cmath.exp(-1.0j*theta_value/2), 0.0], [0.0, cmath.exp(-1.0j*theta_value/2)]], dtype=torch.complex64)
                initial_state_vector = torch.kron(initial_state_vector, psi)

            qubit_count += 1

        return initial_state_vector
    
    def rx_state(self, angle):
        return tensor([[math.cos(angle/2), -1j*math.sin(angle/2)],
                            [-1j*math.sin(angle/2), math.cos(angle/2)]], dtype=torch.complex64)

    def ry_state(self, angle):
        return tensor([[math.cos(angle/2), -math.sin(angle/2)], [math.sin(angle/2), math.cos(angle/2)]], dtype=torch.complex64)
    
    def rz_state(self, angle):
        return tensor([[-cmath.exp(-1.0j*angle/2), 0.0], [0.0, cmath.exp(-1.0j*angle/2)]], dtype=torch.complex64)
    
    rotation_function = {
        'rx': rx_state,
        'ry': ry_state,
        'rz': rz_state
    }
    
    def randomAngleEmbedding(self, angle_tensor):
        rotation_gate_options = ['rx', 'ry', 'rz']
        angle_count = 0
        eye_tensor = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        eye_tensor[0] = 1.0
        for angle in angle_tensor:
            random.seed(520)
            chosen_gate = random.choice(rotation_gate_options)
            if angle_count == 0:
                state_vector = self.rotation_function[chosen_gate](self, angle)
            else:
                new_vector = self.rotation_function[chosen_gate](self, angle)
                state_vector = torch.kron(state_vector, new_vector)
            angle_count+=1
            
        return state_vector @ eye_tensor

    def uniformAngleEmbedding(self, patch_vector, rotation_gate: str):
        eye_tensor = torch.zeros(2**self.n_qubits, dtype=torch.complex64)
        eye_tensor[0] = 1.0
        if rotation_gate == 'rx':
            return self.rx_rotation_states(patch_vector) @ eye_tensor

        elif rotation_gate == 'ry':
            return self.ry_rotation_states(patch_vector) @ eye_tensor

        elif rotation_gate == 'rz':
            return self.rz_rotation_states(patch_vector)  @ eye_tensor
        

    # Instead of simulating one qubit gate per time just do kronecker product to create a layer of quantum gates
    # To link all layers just do matmul

    # Also for simulating a controlled not gate, just do the tensor product between dims=([2,3], [control, target])
    # Tensor product between gates of the same layer would be better tbh, so just do

    def simulate_one_qubit_gate(self, selected_gate: torch.Tensor, state_vector: torch.Tensor, selected_qubit:int) -> tensor:
        return tensordot(selected_gate, state_vector, dims=([1],[selected_qubit]))
    
    def simulate_cnot(self, state_vector: torch.Tensor, target_qubit: int, control_qubit: int) -> tensor:
        return tensordot(self.cnot_gate, state_vector, dims=([2,3],[control_qubit, target_qubit]))

    # Fix pauli Z measurement
    def pauliZ_test(self, state_vector, qubit_index):
        bitmask = 1 << qubit_index
        conjugated_state_vector = torch.conj(state_vector)

        for index in range(0, 2**self.n_qubits):
            if index & bitmask != 0:
                state_vector[index] = -state_vector[index]

        return torch.sum(conjugated_state_vector * state_vector)
    
    def pauliZ_expectationValue(self, state_vector, qubit_index):
        indices = torch.arange(2**self.n_qubits)
        signs = 1 - 2*((indices >> qubit_index) & 1)  # +1 for 0, -1 for 1
        expectation = torch.sum(signs * torch.abs(state_vector)**2)
        
        return expectation


    def simulate_circuit(self, state_vector: torch.Tensor, ansatz_chromosome: list) -> float:
        layer_count = 0
        for layer in ansatz_chromosome:
            print(f'Starting simulation at layer #{layer_count}')
            qubit_count = 0
            cnot_stack = []
            for gate in layer:
                print(f'Starting simulation at qubit {qubit_count}... ')
                if gate in self.gate_operation:
                    state_vector = self.simulate_one_qubit_gate(self.gate_operation[gate], state_vector, qubit_count)
                elif gate is not None:
                    if gate[0:4] == 'ctrl' or gate[0:4] == 'trgt':
                        if not cnot_stack:
                            cnot_stack.append([gate[0:4], qubit_count])
                        else:
                            if gate[0:4] == 'trgt':
                                target_qubit = qubit_count
                                gate_name, control_qubit = cnot_stack[-1]
            
                            if gate[0:4] == 'ctrl':
                                control_qubit = qubit_count
                                gate_name, target_qubit = cnot_stack[-1]

                            state_vector = self.simulate_cnot(state_vector, target_qubit, control_qubit)
                            cnot_stack.pop()

                print(f'Simulation at qubit {qubit_count} done!')
                qubit_count+=1
            layer_count+=1

        
        return [self.pauliZ_expectationValue(state_vector, qubit_index) for qubit_index in range(self.n_qubits)]
    