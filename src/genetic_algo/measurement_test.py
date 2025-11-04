from ansatz_simulation_class import AnsatzSimulation
import torch

def ansatz_simulation_test():
    ansatz_test = AnsatzSimulation(n_qubits=4)
    psi_test_vector = torch.tensor(data=[0.0913-0.1182j,
                                    -0.0627+0.2476j,
                                    0.0491+0.0425j,
                                    -0.0382-0.1801j,
                                    -0.0418-0.1340j,
                                    0.2152-0.0218j,
                                    -0.0735+0.0433j,
                                    0.1442+0.0497j,
                                    0.1283-0.2044j,
                                    0.0018+0.1007j,
                                    -0.1793-0.0908j,
                                    0.0535-0.0261j,
                                    0.0911+0.1538j,
                                    -0.0836+0.0905j,
                                    0.0599-0.0783j,
                                    -0.0974+0.0875j], dtype=torch.complex64)
    print(ansatz_test.measure_state(psi_test_vector))
    
if __name__ == '__main__':
    ansatz_simulation_test()