import numpy as np
import random

def save_generation(gen_number, population, scores):
    file_path = 'circuits/mnist/experiment_#01'
    try:
    # Open the file in write mode ('w') and assign it to the variable 'file'
        with open(file_path, 'w') as file:
            # Use the write() method to put content into the file
            file.write(f'Generation #{gen_number}')
            file.write(population)
            file.write(f'Gen #{gen_number} Scores')
            file.write(scores)
        print(f"Successfully saved text to {file_path}")
    except IOError as e:
        print(f"Error saving file: {e}")

class GeneticAlgorithm():
    def __init__(self, crossover_rate: float, mutation_rate: float, no_generations: int, population_size: int, sampling, fitness, crossover, mutation, duplicates):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.no_generations = no_generations
        self.population_size = population_size

        self.Sampling = sampling
        self.Fitness = fitness
        self.Crossover = crossover
        self.Mutation = mutation
        self.eliminate_duplicates = duplicates

    ## Population deve ter dimensões (population_size, solution_size)

    def initialize_population(self):
        return self.Sampling._do(self.population_size)
  
    def define_fitness(self, population, iteration) -> np.array:
        return np.array([self.Fitness._evaluate(individual, iteration, population.index(individual)) for individual in population])

    def duplicate_threshold(self, offspring_a, offspring_b):
        if self.eliminate_duplicates.is_equal(offspring_a, offspring_b):
            drop_index = random.randint(len(offspring_b))
            offspring_b.pop(drop_index)
        
        return offspring_a, offspring_b
    
    def select_parents(self, population: list):

        random_generator = np.random.default_rng(seed=42)
        indices = random_generator.choice(len(population), 2, replace=False)
        parent_a = population[indices[0]]
        parent_b = population[indices[1]]
        
        if len(parent_a) < len(parent_b):
            parent_a, parent_b = parent_b, parent_a
        return parent_a, parent_b

    def choose_parent(self) -> int:
        options = [0, 1]
        return random.choice(options)
    
    def crossover(self, parent_a, parent_b):
        return self.Crossover._do(parent_a, parent_b)

    def mutate_offspring(self, offspring) -> np.array:
        return self.Mutation._do(offspring)
    
    ## Use fitness values for elitism or something


    def run_algorithm(self) -> tuple:
        iterations = 1
        old_population = self.initialize_population()
        all_generations = []
        iteration_axis = []
        best_chromossome_axis = []
        while iterations != self.no_generations:
            
            print(old_population)   
            print(f"iteration #{iterations}")
            iteration_axis.append(iterations)
            old_fitness_values = self.define_fitness(old_population, iterations)
            save_generation(iterations, old_population, old_fitness_values)
            new_population = []
            for individuals in range(int(self.population_size/2)):
                parent_a, parent_b = self.select_parents(old_population)
                offspring_a, offspring_b = self.mutate_offspring(self.crossover(parent_a, parent_b))
                
                new_population.append(offspring_a)
                new_population.append(offspring_b)
            
            iterations +=1

            old_best_individual_position = np.argmax(old_fitness_values)
            old_best_fitness_value = np.max(old_fitness_values)
            old_best_individual = old_population[old_best_individual_position]

            new_fitness_values = self.define_fitness(new_population, iterations)
            new_best_individual_position = np.argmax(new_fitness_values)
            new_best_fitness_value = np.max(new_fitness_values)
            new_best_individual = new_population[new_best_individual_position]
            print(new_population)
        
            current_best_individual_fitness = 0.0
            if new_best_fitness_value > old_best_fitness_value:
                current_best_individual_fitness = new_best_fitness_value
                current_best_individual = new_best_individual
            else:
                current_best_individual_fitness = old_best_fitness_value
                current_best_individual = old_best_individual

    
            result_tuple = (current_best_individual, current_best_individual_fitness)
            best_chromossome_axis.append(current_best_individual_fitness)
        
            old_best_individual_position = np.argmax(old_fitness_values)
            random_position = random.randint(0, self.population_size - 1)
            new_population[random_position] = old_population[old_best_individual_position]
            old_population = new_population
            
        
        return result_tuple, best_chromossome_axis, iteration_axis, all_generations