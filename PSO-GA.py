import matplotlib.pyplot as plt
import numpy as np
import random

# Parameters (combination of PSO and GA parameters)
num_particles = 30             # Number of particles in the swarm
num_dimensions = 2             # Dimensionality of the problem
max_iterations = 100           # Maximum number of iterations
w = 0.5                        # Inertia weight (PSO)
c1 = 1.5                       # Personal acceleration coefficient (PSO)
c2 = 1.5                       # Social acceleration coefficient (PSO)
bounds = (-5.12, 5.12)         # Bounds for the search-space
crossover_rate = 0.8           # Probability of performing crossover (GA)
mutation_rate = 0.1            # Probability of performing mutation (GA)

# Define the Fitness Function
def rastrigin_function(x):
    n = len(x)
    return 10 * n + sum([xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x])

# Define the Particle Class with GA Operators
class HybridParticle:
    def __init__(self):
        self.position = np.array([random.uniform(bounds[0], bounds[1]) for _ in range(num_dimensions)])
        self.velocity = np.array([random.uniform(-abs(bounds[1] - bounds[0]), abs(bounds[1] - bounds[0])) for _ in range(num_dimensions)])
        self.best_position = self.position.copy()
        self.best_fitness = rastrigin_function(self.position)

    def update_velocity(self, global_best_position):
        r1 = np.random.rand(num_dimensions)
        r2 = np.random.rand(num_dimensions)
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def update_position(self):
        self.position += self.velocity
        self.position = np.maximum(self.position, bounds[0])  # Ensure position is within bounds
        self.position = np.minimum(self.position, bounds[1])

    def mutate(self):
        for i in range(num_dimensions):
            if random.random() < mutation_rate:
                self.position[i] += np.random.normal(0, 1)
                self.position[i] = np.clip(self.position[i], bounds[0], bounds[1])  # Ensure bounds

    @staticmethod
    def crossover(parent1, parent2):
        child1 = HybridParticle()
        child2 = HybridParticle()
        if random.random() < crossover_rate:
            point = random.randint(1, num_dimensions - 1)
            child1.position = np.concatenate((parent1.position[:point], parent2.position[point:]))
            child2.position = np.concatenate((parent2.position[:point], parent1.position[point:]))
        else:
            child1.position = parent1.position.copy()
            child2.position = parent2.position.copy()
        child1.best_position = child1.position.copy()
        child1.best_fitness = rastrigin_function(child1.position)
        child2.best_position = child2.position.copy()
        child2.best_fitness = rastrigin_function(child2.position)
        return child1, child2

# Initialize the Swarm
swarm = [HybridParticle() for _ in range(num_particles)]
global_best_position = swarm[0].best_position.copy()
global_best_fitness = swarm[0].best_fitness

# Initialize list to store global best fitness
best_fitness_progress = []

# Hybrid Algorithm Main Loop
for iteration in range(max_iterations):
    for particle in swarm:
        fitness = rastrigin_function(particle.position)
        if fitness < particle.best_fitness:  # Update personal best
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()
        if fitness < global_best_fitness:  # Update global best
            global_best_fitness = fitness
            global_best_position = particle.position.copy()

    # Append the global best fitness to the progress tracker
    best_fitness_progress.append(global_best_fitness)

    # Apply genetic operations
    offspring = []
    for i in range(0, num_particles, 2):
        parent1 = swarm[i]
        parent2 = swarm[(i+1) % num_particles]
        child1, child2 = HybridParticle.crossover(parent1, parent2)  # Crossover
        child1.mutate()  # Mutation
        child2.mutate()
        offspring.extend([child1, child2])

    for particle in offspring:
        particle.update_velocity(global_best_position)
        particle.update_position()

    swarm = offspring.copy()  # Replace old swarm with offspring
    print(f"Iteration {iteration+1}/{max_iterations}, Global Best Fitness: {global_best_fitness}")

print("\nOptimal Solution:")
print(f"Position: {global_best_position}")
print(f"Fitness: {global_best_fitness}")

# Plotting the fitness over iterations
plt.plot(range(1, max_iterations + 1), best_fitness_progress)
plt.xlabel('Iteration')
plt.ylabel('Global Best Fitness')
plt.title('Hybrid PSO-GA Progress Over Iterations')
plt.show()
