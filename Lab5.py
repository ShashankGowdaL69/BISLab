import numpy as np

def gwo(objective_function, n, dim, max_gen, lb, ub):
    wolves = np.random.uniform(lb, ub, (n, dim))
    fitness = np.apply_along_axis(objective_function, 1, wolves)
    alpha, beta, delta = np.copy(wolves), np.copy(wolves), np.copy(wolves)
    alpha_f, beta_f, delta_f = np.copy(fitness), np.copy(fitness), np.copy(fitness)
    
    for gen in range(max_gen):
        sorted_idx = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_idx[0]], wolves[sorted_idx[1]], wolves[sorted_idx[2]]
        alpha_f, beta_f, delta_f = fitness[sorted_idx[0]], fitness[sorted_idx[1]], fitness[sorted_idx[2]]
        
        A = 2 * np.random.rand(n, dim) - 1
        C = 2 * np.random.rand(n, dim)
        for i in range(n):
            D_alpha = np.abs(C[i] * alpha - wolves[i])
            D_beta = np.abs(C[i] * beta - wolves[i])
            D_delta = np.abs(C[i] * delta - wolves[i])
            
            wolves[i] = wolves[i] + A[i] * (D_alpha + D_beta + D_delta) / 3
            wolves[i] = np.clip(wolves[i], lb, ub)
        
        fitness = np.apply_along_axis(objective_function, 1, wolves)
        print(f"Gen {gen+1}/{max_gen}, Best Fitness: {min(fitness)}")
    
    return alpha, alpha_f

def sphere_function(x):
    return np.sum(x**2)

n = 30
dim = 5
max_gen = 50
lb, ub = -5, 5

best_solution, best_fitness = gwo(sphere_function, n, dim, max_gen, lb, ub)
print("\nBest Solution Found: ", best_solution)
print("Best Fitness: ", best_fitness)
