
import numpy as np
import random


#### *Define PSO param
# PSO Parameters
num_particles = 30            # Number of particles in the swarm
num_dimensions = 2            # Dimensionality of the problem
max_iterations = 100          # Maximum number of iterations to run
w = 0.5                       # Inertia weight
c1 = 1.5                      # Personal acceleration coefficient
c2 = 1.5                      # Social acceleration coefficient
bounds = (-5.0, 5.0)          # Bounds for the search-space


#### *the Fitness Function*


def rosenbrock_function(x):
    return sum([100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1)])


class Particle:
    def __init__(self):
        self.position = np.array([random.uniform(bounds[0], bounds[1]) for _ in range(num_dimensions)])
        self.velocity = np.array([random.uniform(-abs(bounds[1] - bounds[0]), abs(bounds[1] - bounds[0])) for _ in range(num_dimensions)])
        self.best_position = self.position.copy()
        self.best_fitness = rosenbrock_function(self.position)

    def update_velocity(self, global_best_position):
        r1 = np.random.rand(num_dimensions)
        r2 = np.random.rand(num_dimensions)
        cognitive_velocity = c1 * r1 * (self.best_position - self.position)
        social_velocity = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive_velocity + social_velocity

    def update_position(self):
        self.position += self.velocity
        # Ensure position is within bounds
        self.position = np.maximum(self.position, bounds[0])
        self.position = np.minimum(self.position, bounds[1])


#### *Initialize the Swarm*

swarm = [Particle() for _ in range(num_particles)]
global_best_position = swarm[0].position.copy()
global_best_fitness = swarm[0].best_fitness


#### *PSO Main Loop*
for iteration in range(max_iterations):
    for particle in swarm:
        fitness = rosenbrock_function(particle.position)
        
        # Update personal best
        if fitness < particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()
            
        # Update global best
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particle.position.copy()
    
    # Update velocities and positions
    for particle in swarm:
        particle.update_velocity(global_best_position)
        particle.update_position()
    
    # Output current progress
    print(f"Iteration {iteration+1}/{max_iterations}, Global Best Fitness: {global_best_fitness}")

print("\nOptimal Solution:")
print(f"Position: {global_best_position}")
print(f"Fitness: {global_best_fitness}")



import matplotlib.pyplot as plt
from matplotlib import animation

# Prepare figure
fig, ax = plt.subplots()
particles_positions = []

# Collect positions for each iteration
for iteration in range(max_iterations):
    current_positions = []
    for particle in swarm:
        fitness = rosenbrock_function(particle.position)
        if fitness < particle.best_fitness:
            particle.best_fitness = fitness
            particle.best_position = particle.position.copy()
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particle.position.copy()
        current_positions.append(particle.position.copy())
    particles_positions.append(current_positions)
    for particle in swarm:
        particle.update_velocity(global_best_position)
        particle.update_position()

# Animation function
def animate(i):
    ax.clear()
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    positions = particles_positions[i]
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    ax.scatter(x_coords, y_coords, color='blue')
    ax.set_title(f"Iteration {i+1}")

anim = animation.FuncAnimation(fig, animate, frames=max_iterations, interval=200)
plt.show()
