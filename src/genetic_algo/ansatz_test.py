from ansatz_simulation_class import AnsatzSimulation
import torch
import math
import time

def main():
    n_qubits = 4
    angle_tensor = torch.tensor([1.9835, 1.9835, 0.45, 1.8603], dtype=torch.float64)
    chromossome = [['ctrl_0', 'trgt_0', 'ctrl_1', 'trgt_1'], [None, 'ctrl_0', 'trgt_0', None], ['rz_gate', 'pauli_y', 'rx_gate', 'phase']]
    my_ansatz = AnsatzSimulation(n_qubits)
    start_time = time.perf_counter()
    measurement = my_ansatz.simulate_circuit(angle_tensor, 'rz', chromossome)
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"The ansatz simulation executed in {duration:.4f} seconds")
    print(measurement)
   
    
if __name__ == '__main__':
    main()