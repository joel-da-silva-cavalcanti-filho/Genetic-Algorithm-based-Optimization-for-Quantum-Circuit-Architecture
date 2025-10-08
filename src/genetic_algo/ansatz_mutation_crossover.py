from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
import numpy as np
import random

class AnsatzMutation(Mutation):
    def __init__(self, mutation_rate: float):
        super().__init__()
        self.mutation_rate = mutation_rate  

    def _do(self, problem, X, **kwargs):

        for individual in range(len(X)):

            random_value = np.random.random()

            if random_value < self.mutation_rate:
                i = 0
            

class AnsatzCrossover(Crossover):
    def __init__(self):

        super().__init__(2,2)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape

        # The output owith the shape (n_offsprings, n_matings, n_var)
        # Because there the number of parents and offsprings are equal it keeps the shape of X
        Y = np.full_like(X, None, dtype=object)

        # for each mating provided
        for k in range(n_matings):

            # get the first and the second parent
            a, b = X[0, k, 0], X[1, k, 0]

            # prepare the offsprings
            off_a = ["_"] * problem.n_characters
            off_b = ["_"] * problem.n_characters

            for i in range(problem.n_characters):
                if np.random.random() < 0.5:
                    off_a[i] = a[i]
                    off_b[i] = b[i]
                else:
                    off_a[i] = b[i]
                    off_b[i] = a[i]

            # join the character list and set the output
            Y[0, k, 0], Y[1, k, 0] = "".join(off_a), "".join(off_b)

        return Y