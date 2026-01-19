from ansatz_optimization_problem import QuantumCircuitOptimization, QuantumCircuitSampling
from ansatz_mutation_crossover import AnsatzCrossover, AnsatzMutation, RemoveEquivalentAnsätze
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from torchvision import datasets, transforms
from patch_making import PatchExtraction, sample_data, SampledDataset4Training
from torch.utils.data import Dataset, DataLoader, Subset
import pickle
from ga import GeneticAlgorithm

if __name__ == '__main__':
    print('starting the experiment!!')
    # Genetic Algorithm hyperparameters
    population_size = 2
    crossover_rate = 0.5
    mutation_rate = 0.1
    generations = 2
    seed = 42
    
    # Circuit ansatz hyperparameters
    n_tests = 4
    equivalence_ratio = 0.7
    n_qubits = 4
    max_gates = 60
    gate_options = [None, 'pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard', 'ctrl', 'trgt']
    max_layers = 10
    
    # Model and training hyperparameters
    input_size = 20
    n_classes = 2
    mode = 'frozen'
    batch_size = 50
    train_ratio = 0.7
    patch_size = 2
    
    
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        PatchExtraction(patch_size)])
    
    target_classes = [0, 1]
    sample_size = 200
    
    full_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)


    sampled_training_dataset = sample_data(full_trainset, target_classes, sample_size)
    mnist_training_dataset = SampledDataset4Training(sampled_training_dataset, target_classes)


    ansatz_optimization_problem = QuantumCircuitOptimization(n_qubits=n_qubits,
                                                             n_gates=max_gates,
                                                             possible_gates=gate_options,
                                                             n_layers=max_layers,
                                                             patch_size=patch_size,
                                                             max_gates=max_gates,
                                                             input_size=input_size,
                                                             n_classes=n_classes,
                                                             mode=mode,
                                                             dataset=mnist_training_dataset,
                                                             batch_size=batch_size,
                                                             train_ratio=train_ratio)
    
    duplicate_ansatz = RemoveEquivalentAnsätze(n_tests, equivalence_ratio, n_qubits)
    
    genetic_algo = GeneticAlgorithm(crossover_rate=crossover_rate,
                                    mutation_rate=mutation_rate,
                                    no_generations=generations,
                                    population_size=population_size,
                                    sampling=QuantumCircuitSampling(n_qubits),
                                    fitness=ansatz_optimization_problem,
                                    crossover=AnsatzCrossover(crossover_rate),
                                    mutation=AnsatzMutation(mutation_rate),
                                    duplicates=duplicate_ansatz)
    
    result, best_solution_axis, iterations = genetic_algo.run_algorithm()

    
    #genetic_algo.setup(ansatz_optimization_problem, ('n_gen', generations), verbose=True, progress=True, seed=seed)
