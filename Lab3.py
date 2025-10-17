import numpy as np

def ant_colony_optimization(dist_matrix, n_ants, n_iterations, alpha=1, beta=2, rho=0.5, q=100):
    n = len(dist_matrix)
    pheromone = np.ones((n, n))
    visibility = 1 / (dist_matrix + np.eye(n) * 1e10)
    best_length = np.inf
    best_path = None
    
    for _ in range(n_iterations):
        all_paths = []
        all_lengths = []
        for _ in range(n_ants):
            path = [np.random.randint(n)]
            while len(path) < n:
                i = path[-1]
                prob = (pheromone[i] ** alpha) * (visibility[i] ** beta)
                prob[path] = 0
                prob /= prob.sum()
                next_city = np.random.choice(range(n), p=prob)
                path.append(next_city)
            length = sum(dist_matrix[path[i], path[i+1]] for i in range(n-1)) + dist_matrix[path[-1], path[0]]
            all_paths.append(path)
            all_lengths.append(length)
            if length < best_length:
                best_length = length
                best_path = path
        pheromone *= (1 - rho)
        for path, length in zip(all_paths, all_lengths):
            for i in range(n-1):
                pheromone[path[i], path[i+1]] += q / length
            pheromone[path[-1], path[0]] += q / length
    return best_path, best_length

dist = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])

path, length = ant_colony_optimization(dist, n_ants=10, n_iterations=100)
print("Best path:", path)
print("Best length:", length)
