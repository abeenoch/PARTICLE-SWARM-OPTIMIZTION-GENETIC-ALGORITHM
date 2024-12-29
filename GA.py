import numpy as np
import random

# GA Parameters
population_size = 50          # Number of individuals in the population
num_generations = 100         # Maximum number of generations
num_dimensions = 2            # Dimensionality of the problem
crossover_rate = 0.8          # Probability of performing crossover
mutation_rate = 0.1           # Probability of performing mutation
bounds = (-5.12, 5.12)        # Bounds for the search-space
elitism = True                # Whether to carry the best individual to the next generation

def rastrigin_function(x):
    n = len(x)
    return 10 * n + sum([xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x])

#### **Chromosome Representation**


class Individual:
    def __init__(self):
        self.chromosome = np.array([random.uniform(bounds[0], bounds[1]) for _ in range(num_dimensions)])
        self.fitness = rastrigin_function(self.chromosome)

    def mutate(self):
        for i in range(num_dimensions):
            if random.random() < mutation_rate:
                # Gaussian mutation
                self.chromosome[i] += np.random.normal(0, 1)
                # Ensure the gene is within bounds
                self.chromosome[i] = np.clip(self.chromosome[i], bounds[0], bounds[1])
        self.fitness = rastrigin_function(self.chromosome)


def create_initial_population():
    return [Individual() for _ in range(population_size)]



def tournament_selection(population, k=3):
    selected = random.sample(population, k)
    best = min(selected, key=lambda ind: ind.fitness)
    return best

def single_point_crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, num_dimensions - 1)
        child1_chromosome = np.concatenate((parent1.chromosome[:point], parent2.chromosome[point:]))
        child2_chromosome = np.concatenate((parent2.chromosome[:point], parent1.chromosome[point:]))
    else:
        child1_chromosome = parent1.chromosome.copy()
        child2_chromosome = parent2.chromosome.copy()
    child1 = Individual()
    child2 = Individual()
    child1.chromosome = child1_chromosome
    child2.chromosome = child2_chromosome
    child1.fitness = rastrigin_function(child1.chromosome)
    child2.fitness = rastrigin_function(child2.chromosome)
    return child1, child2

#### **The GA Main Loop**

# Creationte initial popula
population = create_initial_population()

# Track the best solution
best_individual = min(population, key=lambda ind: ind.fitness)

for generation in range(num_generations):
    new_population = []

    # Elitism: carry the best individual to the next generation
    if elitism:
        new_population.append(best_individual)

    while len(new_population) < population_size:
        # Selection
        parent1 = tournament_selection(population)
        parent2 = tournament_selection(population)

        # Crossover
        child1, child2 = single_point_crossover(parent1, parent2)

        # Mutation
        child1.mutate()
        child2.mutate()

        # Add children to the new population
        new_population.extend([child1, child2])

    # Truncate if we have an extra individual
    population = new_population[:population_size]

    # Update the best individual
    current_best = min(population, key=lambda ind: ind.fitness)
    if current_best.fitness < best_individual.fitness:
        best_individual = current_best

    # Output current progress
    print(f"Generation {generation+1}/{num_generations}, Best Fitness: {best_individual.fitness}")

print("\nOptimal Solution:")
print(f"Chromosome: {best_individual.chromosome}")
print(f"Fitness: {best_individual.fitness}")



