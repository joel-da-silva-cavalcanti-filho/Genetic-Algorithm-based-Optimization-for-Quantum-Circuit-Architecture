from ansatz_simulation_class import AnsatzSimulation
import numpy as np
import torch
from torch import tensor, tensordot

class ParametrizedAnsatz(AnsatzSimulation):
    def __init__(self, n_qubits: int):
        super().__init__(n_qubits)
        self.n_qubits = n_qubits
        
        

    def create_parameter_list(self, no_parameters: int):
        return np.random.rand(no_parameters)

    def calculate_number_parameters(self, ansatz_chromosome: list) -> int:
        angle_count = 0
        for layer in ansatz_chromosome:
            for gate in layer:
                if gate == 'rx' or gate == 'ry' or gate == 'rz':
                    angle_count += 1
                    
        return angle_count
    
    def simulate_rotation_gate(self, selected_gate, state_vector, selected_qubit, angle):
        rotation_tensor = self.rotation_function[selected_gate](self, angle)
        return tensordot(rotation_tensor, state_vector, dims=([1],[selected_qubit]))
    
    
    def simulate_circuit(self, state_vector: torch.Tensor, ansatz_chromosome: list, parameters: np.array) -> float:
        layer_count = 0
        parameter_count = 0
        for layer in ansatz_chromosome:
            print(f'Starting simulation at layer #{layer_count}')
            qubit_count = 0
            cnot_stack = []
            for gate in layer:
                print(f'Starting simulation at qubit {qubit_count}... ')
                if gate in self.rotation_function:
                    state_vector = self.simulate_rotation_gate(gate, state_vector, qubit_count, parameters[parameter_count])
                    parameter_count+=1
                elif gate in self.non_parametrized_gates:
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

        return self.measureState(state_vector)