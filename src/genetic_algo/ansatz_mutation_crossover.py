from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.duplicate import ElementwiseDuplicateElimination
import numpy as np
import random
from mutation_operators import swap_cnot_gate, change_cnot_gate_to_one_qubit_gates

# Gotta test AnsatzMutation

class AnsatzMutation(Mutation):

    """
    This Mutation Class randomly swaps Ansatz layers and changes the  
    the gates placed in certain positions

    """
    def __init__(self, mutation_rate: float):
        super().__init__()
        self.mutation_rate = mutation_rate  
        self.one_qubit_gates = [None, 'pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard']
        self.gate_options_with_cnot = [None, 'pauli_x', 'pauli_y','pauli_z','rx_gate','ry_gate','rz_gate','phase','t','hadamard', 'ctrl', 'trgt']

    def _do(self, problem, X, **kwargs):

        for individual in range(len(X)):

            for circuit_layer in individual:
                random_value = np.random.random()
                if random_value < self.mutation_rate:
                    random_qubit_position = np.random.randint(0, len(circuit_layer))
                    random_gate =  circuit_layer[random_qubit_position]
                    if random_gate[0:4] == 'ctrl' or random_gate[0:4] == 'trgt': 
                        random_mutation_action = np.random.randint((0,3))
                        random_qubit_position = np.random.randint(0, len(circuit_layer))
                        if random_mutation_action == 0:
                            circuit_layer = swap_cnot_gate(circuit_layer)
                        elif random_mutation_action == 1:
                            circuit_layer = change_cnot_gate_to_one_qubit_gates(circuit_layer, self.one_qubit_gates, random_qubit_position)
                    else:
                        # Arrumar para não gerar mutações não-válidas
                        circuit_layer[random_qubit_position] = random.choice(self.gate_options_with_cnot)

        return X
    
def select_parents(population: list):
    random_generator = np.random.default_rng(seed=42)
    parent_a, parent_b = random_generator.choice(a = np.array(population), size=2, replace=False)
    if parent_a.shape[1] < parent_b.shape[1]:
        parent_a, parent_b = parent_b, parent_a
    return parent_a, parent_b


# Gotta test AnsatzCrossover
class AnsatzCrossover(Crossover):
    """"
    The crossover happens between chromosomes of possibly different lengths,
    then the offspring might inherit the size of one of its parents
    """

    def __init__(self, crossover_rate: float):

        super().__init__(2,2)
        self.crossover_rate = crossover_rate

    def _do(self, problem, X, **kwargs):

        random_generator = np.random.default_rng(seed=42)
        n_matings = X.shape[0]
        offsprings = []

        # Should i preserve the biggest length of chromossomes or not? Let's do it randomly
        for mating in range(n_matings):
            
            parent_a, parent_b = select_parents(population=X)

            # The parent b is always the smallest one, and the parent a, the biggest one
            chromosome_length_a = parent_b.shape[1]
            chromosome_length_b = parent_a.shape[1]

            crossover_point_a = random_generator.randint((1, chromosome_length_a))
            crossover_point_b = random_generator.randint((1, chromosome_length_b))

            # Could flip a coin to change which parent comes first
            flip_coin = random_generator.choice([0, 1])

            if flip_coin == 0:
                offspring_a = np.concatenate((parent_a[0:crossover_point_a], parent_b[crossover_point_b]), axis=0)
                offspring_b = np.concatenate((parent_b[0:crossover_point_b], parent_a[crossover_point_a:]), axis=0)
            else:
                offspring_a = np.concatenate((parent_b[0:crossover_point_b], parent_a[crossover_point_a:]), axis=0)
                offspring_b = np.concatenate((parent_a[0:crossover_point_a], parent_b[crossover_point_b]), axis=0)
            
            offsprings.append(offspring_a)
            offsprings.append(offspring_b)
        
        return offsprings
       # _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        #Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        #for k in range(n_matings):

            # get the first and the second parent
        #    a, b = X[0, k, 0], X[1, k, 0]

            # prepare the offsprings
            #off_a = ["_"] * problem.n_characters
            #off_b = ["_"] * problem.n_characters

            #for i in range(problem.n_characters):
            #   if np.random.random() < 0.5:
            #        off_a[i] = a[i]
             #       off_b[i] = b[i]
             #   else:
             #       off_a[i] = b[i]
             #       off_b[i] = a[i]

            # join the character list and set the output
            #Y[0, k, 0], Y[1, k, 0] = "".join(off_a), "".join(off_b)

        #return Y
    
    class RemoveEquivalentAnsätze(ElementwiseDuplicateElimination):

        def is_equal(self, ansatz_a, ansatz_b):
            return False