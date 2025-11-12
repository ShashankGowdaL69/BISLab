import random

# Parameters
POP_SIZE = 20
GENOME_LENGTH = 12
GENERATIONS = 25
MUTATION_RATE = 0.15

# Fitness function: maximize number of 1's
def fitness(individual):
    return sum(individual)

# Initialize population randomly
population = [[random.randint(0, 1) for _ in range(GENOME_LENGTH)] for _ in range(POP_SIZE)]

# Tournament selection from neighbors
def tournament_selection(pop, fit_vals, idx, neighbors):
    best = idx
    for n in neighbors:
        if fit_vals[n] > fit_vals[best]:
            best = n
    return pop[best][:]

# Single-point crossover
def crossover(p1, p2):
    point = random.randint(1, GENOME_LENGTH - 1)
    return p1[:point] + p2[point:]

# Mutation
def mutate(individual):
    for i in range(GENOME_LENGTH):
        if random.random() < MUTATION_RATE:
            individual[i] = 1 - individual[i]

# 1D ring topology
def get_neighbors(i):
    left = (i - 1) % POP_SIZE
    right = (i + 1) % POP_SIZE
    return [left, right]

# Evolution process
for gen in range(GENERATIONS):
    fitness_values = [fitness(ind) for ind in population]
    new_population = []

    for i in range(POP_SIZE):
        neighbors = get_neighbors(i)
        p1 = tournament_selection(population, fitness_values, i, neighbors)
        p2 = tournament_selection(population, fitness_values, i, neighbors)
        child = crossover(p1, p2)
        mutate(child)
        new_population.append(child)

    population = new_population

    best_fit = max(fitness_values)
    avg_fit = sum(fitness_values) / len(fitness_values)
    print(f"Generation {gen + 1:02d}: Best = {best_fit}, Avg = {avg_fit:.2f}")

# Final results
final_fitness = [fitness(ind) for ind in population]
best_index = final_fitness.index(max(final_fitness))
best_solution = population[best_index]

print("\nBest Solution:", best_solution)
print("Best Fitness:", max(final_fitness))
