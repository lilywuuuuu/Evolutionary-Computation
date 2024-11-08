import random
import numpy as np
from deap import base, creator, tools, algorithms

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
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", n_dimensional_sphere)

# (1+1)-ES with fixed step-size Gaussian mutation
def one_plus_one_es(population, toolbox, sigma):
    # clone the parent
    offspring = toolbox.clone(population[0])

    # mutate the offspring
    for i in range(len(offspring)):
        offspring[i] += random.gauss(0, sigma)

    # evaluate the offspring using the n-dimensional sphere model
    offspring.fitness.values = toolbox.evaluate(offspring)

    # if the offspring is better than the parent, replace the parent with the offspring
    if offspring.fitness.values[0] < population[0].fitness.values[0]:
        population[0] = offspring

    return population

# (1,1)-ES with fixed step-size Gaussian mutation
def one_comma_one_es(population, toolbox, sigma):
    # clone the parent
    offspring = toolbox.clone(population[0])

    # mutate the offspring
    for i in range(len(offspring)):
        offspring[i] += random.gauss(0, sigma)
    
    # replace the parent with the offspring
    population[0] = offspring

    return population

def run(test, sigma, runs=10):
    print(f"Running one-{test}-one test with sigma={sigma}")
    for run in range(runs):
        pop = toolbox.population(n=1)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
        
        same_pop_count = 0
        for gen in range(1, 10_000_001):
            if test=='plus':
                new_pop = one_plus_one_es(pop, toolbox, sigma)
            elif test=='comma':
                new_pop = one_comma_one_es(pop, toolbox, sigma)

            if new_pop[0].fitness.values[0] == pop[0].fitness.values[0]:
                same_pop_count += 1
            else:
                same_pop_count = 0
            pop = new_pop

            # the objective value of the individual is equal to or less than 0.005 or the population has stagnated
            if same_pop_count >= 1_000_000:
                break
            if pop[0].fitness.values[0] <= 0.005:
                break
        if same_pop_count == 1_000_000:
            print(f"Run #{run + 1}, Generations: {10_000_000}, Fitness: {pop[0].fitness.values[0]}, Stagnated")
        else:
            print(f"Run #{run + 1}, Generations: {gen}, Fitness: {pop[0].fitness.values[0]}")


if __name__ == '__main__':
    run('plus', 0.01)
    run('plus', 0.1)
    run('plus', 1.0)
    run('comma', 0.01)
    run('comma', 0.1)
    run('comma', 1.0)