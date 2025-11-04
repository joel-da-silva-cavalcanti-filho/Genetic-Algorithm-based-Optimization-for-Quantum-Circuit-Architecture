from ansatz_optimization_problem import QuantumCircuitSampling

def circuit_testing():
    ansatz_sample = QuantumCircuitSampling()
    random_population = ansatz_sample._do(n_qubits=4, n_samples=1)
    print(random_population)
    
if __name__ == '__main__':
    circuit_testing()