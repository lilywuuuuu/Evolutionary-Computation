import numpy as np
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os 

class OneMaxSolver(object):
    def __init__(self, bits:int, num_of_populations :int, num_of_generations:int, cross_probablity: float):
        self.bits = bits
        self.num_of_populations = num_of_populations
        self.num_of_generations = num_of_generations
        self.cross_probablity = cross_probablity
        self.infos = pd.DataFrame(columns=["generation", "max_fitness", "mean_fitness", "min_fitness", "std_fitness"]) 
        self.random_init_populations()
        logging.info(f"Initial populations size: {self.population.shape}")

    def random_init_populations(self):
        self.population = np.random.randint(0 , 2,size = (self.num_of_populations, self.bits))
        self.reocrd_infos(0)

    def reocrd_infos(self, generation):
        fitness = self.evaluate_fitness(self.population)
        new_row = {
            "generation": generation,
            "max_fitness": np.max(fitness),
            "mean_fitness": np.mean(fitness),
            "min_fitness": np.min(fitness),
            "std_fitness": np.std(fitness)
        }
        self.infos.loc[len(self.infos)] = new_row

    def evaluate_fitness(self, population):
        return np.sum(population, axis=1)

    def get_info(self):
        return self.infos

    def roulette_wheel_selection(self, fitness):
        total_fitness = np.sum(fitness)
        probabilities = fitness / total_fitness
        return np.random.choice(len(fitness), p=probabilities)

    def one_point_crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.bits)
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2

    def run(self):
        for generation in range(1, self.num_of_generations + 1):
            fitness = self.evaluate_fitness(self.population)
            new_population = []
            for _ in range(self.num_of_populations // 2):
                parent1 = self.population[self.roulette_wheel_selection(fitness)]
                parent2 = self.population[self.roulette_wheel_selection(fitness)]
                if np.random.rand() < self.cross_probablity:
                    offspring1, offspring2 = self.one_point_crossover(parent1, parent2)
                else:
                    offspring1, offspring2 = parent1, parent2
                new_population.extend([offspring1, offspring2])
            self.population = np.array(new_population)
            self.reocrd_infos(generation)
            
def analyze(infos):
    combined_info = pd.concat(infos)
    mean_info = combined_info.groupby("generation").mean()
    
    if not os.path.exists("q3_output"):
        os.makedirs("q3_output")
    mean_info.to_csv("q3_output/mean_info.csv")
    
    plt.figure(figsize=(12, 7))
    plt.plot(mean_info.index, mean_info["max_fitness"], label="Maximum Fitness", color="red")
    plt.xlabel("Number of Generations")
    plt.ylabel("Maximum Fitness")
    plt.title("Maximum Fitness Across Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig("q3_output/max_fitness.png")
    plt.close()
    
    plt.figure(figsize=(12, 7))
    plt.plot(mean_info.index, mean_info["mean_fitness"], label="Average Fitness", color="green")
    plt.plot(mean_info.index, mean_info["max_fitness"], label="Maximum Fitness", color="red")
    plt.plot(mean_info.index, mean_info["min_fitness"], label="Minimum Fitness", color="blue")
    plt.plot(mean_info.index, mean_info["std_fitness"], label="Std Fitness", color="orange")
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness")
    plt.title("Fitness Across Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig("q3_output/fitness_over_generations.png")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO)
    bits = 50
    num_of_populations = 200
    num_of_generations = 100
    crossover_probablity = 1.0
    runs = 10
    infos = []
    for i in range(runs):
        onemax = OneMaxSolver(bits, num_of_populations, num_of_generations, crossover_probablity)
        onemax.run()
        info = onemax.get_info()
        infos.append(info)
        logging.info(f"Run {i} finished")
    logging.info("All runs finished, start analyzing")
    analyze(infos)
    logging.info("All finished")

if __name__ == "__main__":
    main()
