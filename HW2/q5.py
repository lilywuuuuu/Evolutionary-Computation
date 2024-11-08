import random
import numpy as np
from deap import base, creator, tools

def n_dimensional_sphere(individual):
    return sum(x**2 for x in individual),

# define the individual and fitness
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# create and register the toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=10)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", n_dimensional_sphere)

# constants for uncorrelated mutation with n step-sizes
n = 10
tau = 1 / np.sqrt(2 * np.sqrt(n))
tau_prime = 1 / np.sqrt(2 * n)
epsilon_0 = 1e-6  # minimum step-size threshold to avoid stagnation
omega = 1  # maximum step-size threshold to avoid divergence

# initialize step-sizes for each dimension
def initialize_sigma(initial_sigma):
    return [initial_sigma] * n

# (1+1)-ES with uncorrelated Gaussian mutation with n step-sizes
def one_plus_one_es_uncorrelated(population, toolbox, sigma):
    # clone the parent
    offspring = toolbox.clone(population[0])

    # mutate each step-size
    new_sigma = [
        min(max(s * np.exp(tau_prime * random.gauss(0, 1) + tau * random.gauss(0, 1)), epsilon_0), omega)
        for s in sigma
    ]

    # mutate the individual using the new step-sizes
    for i in range(len(offspring)):
        offspring[i] += random.gauss(0, new_sigma[i])
    
    # evaluate the offspring
    offspring.fitness.values = toolbox.evaluate(offspring)

    # if the offspring is better, replace the parent
    if offspring.fitness.values < population[0].fitness.values:
        population[0] = offspring
        sigma[:] = new_sigma  # update the parent's step-sizes with the new ones

    return population

# (1,1)-ES with uncorrelated Gaussian mutation with n step-sizes
def one_comma_one_es_uncorrelated(population, toolbox, sigma):
    # clone the parent
    offspring = toolbox.clone(population[0])

    # mutate each step-size
    new_sigma = [
        min(max(s * np.exp(tau_prime * random.gauss(0, 1) + tau * random.gauss(0, 1)), epsilon_0), omega)
        for s in sigma
    ]

    # mutate the individual using the new step-sizes
    for i in range(len(offspring)):
        offspring[i] += random.gauss(0, new_sigma[i])

    # replace the parent with the offspring
    population[0] = offspring
    sigma[:] = new_sigma  # update the parent's step-sizes with the new ones

    return population

def run(test, initial_sigma, runs=10):
    print(f"Running one-{test}-one test with uncorrelated mutation, starting σ={initial_sigma}")
    for run in range(runs):
        # initialize population and step-sizes
        pop = toolbox.population(n=1)
        sigma = initialize_sigma(initial_sigma)
        
        # evaluate the initial individual
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        
        pop_same_count = 0
        for gen in range(1, 10_000_001):
            # run the specified ES variant
            if test == 'plus':
                new_pop = one_plus_one_es_uncorrelated(pop, toolbox, sigma)
            elif test == 'comma':
                new_pop = one_comma_one_es_uncorrelated(pop, toolbox, sigma)

            if new_pop[0].fitness.values[0] == pop[0].fitness.values[0]:
                pop_same_count += 1
            else: 
                pop_same_count = 0
            
            # the objective value of the individual is equal to or less than 0.005 or the population has stagnated
            pop = new_pop
            if pop_same_count >= 1_000_000:
                break
            if pop[0].fitness.values[0] <= 0.005:
                break

        if pop_same_count >= 1_000_000:
            print(f"Run #{run + 1}: Generations: {10_000_000}, Fitness: {pop[0].fitness.values[0]}, Stagnated")
        else: 
            print(f"Run #{run + 1}: Generations: {gen}, Fitness: {pop[0].fitness.values[0]}")
    

if __name__ == '__main__':
    run('plus', 0.01)
    run('plus', 0.1)
    run('plus', 1.0)
    run('comma', 0.01)
    run('comma', 0.1)
    run('comma', 1.0)