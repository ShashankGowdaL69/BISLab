import numpy as np
from scipy.special import gamma  # import gamma from scipy.special

def levy_flight(dim, beta=1.5):
    sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
               gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)) ** (1 / beta)
    u = np.random.normal(0, sigma_u, dim)
    v = np.random.normal(0, 1, dim)
    step = u / np.power(np.abs(v), 1 / beta)
    return step

def cuckoo_search(objective_function, n, dim, Pa=0.25, MaxGen=100, bounds=(-5, 5)):
    nests = np.random.uniform(bounds[0], bounds[1], (n, dim))
    fitness = np.apply_along_axis(objective_function, 1, nests)
    best_solution = nests[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    
    for gen in range(MaxGen):
        new_nests = np.copy(nests)
        for i in range(n):
            step = levy_flight(dim)
            new_nests[i] = nests[i] + step
            new_nests[i] = np.clip(new_nests[i], bounds[0], bounds[1])
        new_fitness = np.apply_along_axis(objective_function, 1, new_nests)
        better_nests = new_fitness < fitness
        nests[better_nests] = new_nests[better_nests]
        fitness[better_nests] = new_fitness[better_nests]
        forget_idx = np.random.rand(n) < Pa
        nests[forget_idx] = np.random.uniform(bounds[0], bounds[1], (np.sum(forget_idx), dim))
        fitness = np.apply_along_axis(objective_function, 1, nests)
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = nests[np.argmin(fitness)]
        print(f"Generation {gen + 1}/{MaxGen}, Best Fitness: {best_fitness}")
    return best_solution, best_fitness

def sphere_function(x):
    return np.sum(x**2)

n = 30
dim = 5
MaxGen = 100
Pa = 0.25

best_solution, best_fitness = cuckoo_search(sphere_function, n, dim, Pa, MaxGen)

print("\nBest Solution Found: ", best_solution)
print("Best Fitness Value: ", best_fitness)
