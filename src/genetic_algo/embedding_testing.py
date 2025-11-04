from ansatz_simulation_class import AnsatzSimulation
import torch

def embedding_test():
    ansatz = AnsatzSimulation(n_qubits=4)
    angle_tensor = torch.tensor([1.9835, 1.9835, 1.9096, 1.8603], dtype=torch.float64)
    state_vector = ansatz.randomAngleEmbedding(angle_tensor)
    print(state_vector)
    
if __name__ == '__main__':
    embedding_test()