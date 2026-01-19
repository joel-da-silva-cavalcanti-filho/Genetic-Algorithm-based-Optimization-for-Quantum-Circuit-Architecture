import numpy as np

# Test both functions

def swap_cnot_gate(circuit_layer: list, selected_qubit: int):
    """
    This function swaps the control and target qubits in 
    the CNOT gate

    """

    gate = circuit_layer[selected_qubit]
    if gate[0:4] == 'ctrl': # testar esse if
        cnot_value = gate[5]
        target_index = circuit_layer.index('trgt_'+ cnot_value)
        circuit_layer[selected_qubit], circuit_layer[target_index] =  circuit_layer[target_index], circuit_layer[selected_qubit] 
    elif gate[0:4] == 'trgt':
        cnot_value = gate[5]
        control_index = circuit_layer.index('ctrl_' + cnot_value)
        circuit_layer[selected_qubit], circuit_layer[control_index] =  circuit_layer[control_index], circuit_layer[selected_qubit]
    
    return circuit_layer

def change_cnot_gate_to_one_qubit_gates(circuit_layer: list, gate_options: list, selected_qubit: int) -> list:

    """
    This function turns the control and target qubits into one-qubit gates

    """

    gate = circuit_layer[selected_qubit]
    gate_index = gate[5]
    if gate[0:4] == 'ctrl':
        target_index = circuit_layer.index('trgt_' + gate_index)
        circuit_layer[control_index] = np.random.choice(gate_options)
    elif gate[0:4] == 'trgt':
        control_index = circuit_layer.index('ctrl_' + gate_index)
        circuit_layer[control_index] = np.random.choice(gate_options)
    
    circuit_layer[selected_qubit] = np.random.choice(gate_options)
    
    return circuit_layer