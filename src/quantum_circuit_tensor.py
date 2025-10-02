import torch
import math
import cmath

qubit_0 = torch.tensor([1.0, 1.0],dtype=torch.complex64)

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

def circuit(input_state)