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
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", n_dimensional_sphere)

# constants
G = 50  # Number of generations
a = 1.5  # Scaling factor for step-size adaptation
epsilon_0 = 1e-6  # minimum step-size threshold to avoid stagnation
omega = 1  # Maximum step-size threshold to avoid divergence

# (1+1)-ES with Gaussian mutation and 1/5 rule for step-size adaptation
def one_plus_one_es_15rule(population, toolbox, sigma, G, a):
    successes = 0
    for gen in range(G):
        # clone the parent and mutate the offspring
        offspring = toolbox.clone(population[0])
        for i in range(len(offspring)):
            offspring[i] += random.gauss(0, sigma)

        # evaluate the offspring
        offspring.fitness.values = toolbox.evaluate(offspring)

        # termination condition: objective value <= 0.005
        if offspring.fitness.values[0] <= 0.005:
            population[0] = offspring
            return population, sigma, gen
        
        # check if offspring is better (successful mutation)
        if offspring.fitness.values[0] < population[0].fitness.values[0]:
            population[0] = offspring
            successes += 1  # Count successful mutations

    # apply the 1/5 rule
    success_rate = successes / G
    if success_rate > 0.2:
        sigma *= a  # increase step-size if success rate > 1/5
    else:
        sigma /= a  # decrease step-size if success rate <= 1/5

    return population, sigma, 50

# (1,1)-ES with Gaussian mutation and 1/5 rule for step-size adaptation
def one_comma_one_es_15rule(population, toolbox, sigma, G, a):
    successes = 0
    for gen in range(G):
        # clone the parent and mutate the offspring
        offspring = toolbox.clone(population[0])
        for i in range(len(offspring)):
            offspring[i] += random.gauss(0, sigma)

        # evaluate the offspring
        offspring.fitness.values = toolbox.evaluate(offspring)

        # termination condition: objective value <= 0.005
        if offspring.fitness.values[0] <= 0.005:
            return population, sigma, gen

        # check if offspring is better (successful mutation)
        if offspring.fitness.values[0] < population[0].fitness.values[0]:
            successes += 1  # Count successful mutations

        # replace parent with offspring
        population[0] = offspring  

    # apply the 1/5 rule
    success_rate = successes / G
    if success_rate > 0.2 and sigma < omega:
        sigma *= a  # increase step-size if success rate > 1/5
    elif success_rate <= 0.2 and sigma > epsilon_0:
        sigma /= a  # decrease step-size if success rate <= 1/5

    return population, sigma, 50

def run(test, initial_sigma, G=50, a=1.5, runs=10):
    print(f"Running one-{test}-one test with initial sigma={initial_sigma}")
    for run in range(runs):
        pop = toolbox.population(n=1)
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)

        sigma = initial_sigma  # initialize step-size
        max_gen = int(10_000_000 / G)
        pop_same_count = 0
        for gen in range(0, max_gen):
            if test == 'plus':
                new_pop, sigma, g = one_plus_one_es_15rule(pop, toolbox, sigma, G, a)
            elif test == 'comma':
                new_pop, sigma, g = one_comma_one_es_15rule(pop, toolbox, sigma, G, a)

            if new_pop[0].fitness.values[0] == pop[0].fitness.values[0]:
                pop_same_count += 1
            else:
                pop_same_count = 0
            
            # the objective value of the individual is equal to or less than 0.005 or the population has stagnated
            if pop_same_count >= 1_000_000:
                break
            if pop[0].fitness.values[0] <= 0.005:
                break
        if pop_same_count >= 1_000_000:
            print(f"Run #{run + 1}: Generations: {10_000_000}, Final Sigma: {sigma:.6f}, Fitness: {pop[0].fitness.values[0]}, Stagnated")
        else: 
            print(f"Run #{run + 1}, Generations: {gen * G + g}, Final Sigma: {sigma:.6f}, Fitness: {pop[0].fitness.values[0]}")
        

if __name__ == '__main__':
    run('plus', 0.01)
    run('plus', 0.1)
    run('plus', 1.0)
    run('comma', 0.01)
    run('comma', 0.1)
    run('comma', 1.0)
