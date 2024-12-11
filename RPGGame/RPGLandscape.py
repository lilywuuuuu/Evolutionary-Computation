import pygame
import random
import os

# Initialize pygame
pygame.init()

# Constants
TILE_SIZE = 16
MAP_WIDTH = 40
MAP_HEIGHT = 40
SCREEN_WIDTH = TILE_SIZE * MAP_WIDTH
SCREEN_HEIGHT = TILE_SIZE * MAP_HEIGHT
POPULATION_SIZE = 50
GENERATIONS = 1000
MUTATION_RATE = 0.01
TILE_TYPES = ['0', '1', '2', '3', '4']  # Mountain, River, Grass, Rock, RiverRock

# Constants for the center and radius
center_x = MAP_WIDTH // 2
center_y = MAP_HEIGHT // 2
radius = MAP_HEIGHT // 3
max_distance = ((center_x) ** 2 + (center_y) ** 2) ** 0.5

# Load images
IMAGES = {
    '0': pygame.transform.scale(pygame.image.load('data/mountain.png'), (TILE_SIZE, TILE_SIZE)),
    '1': pygame.transform.scale(pygame.image.load('data/river.png'), (TILE_SIZE, TILE_SIZE)),
    '2': pygame.transform.scale(pygame.image.load('data/grass.png'), (TILE_SIZE, TILE_SIZE)),
    '3': pygame.transform.scale(pygame.image.load('data/rock.png'), (TILE_SIZE, TILE_SIZE)),
    '4': pygame.transform.scale(pygame.image.load('data/riverstone.png'), (TILE_SIZE, TILE_SIZE))
}

# Target distribution for the tiles
target_distribution = {
    '0': 0.35,  # 42% mountains
    '1': 0.26,  # 14% rivers
    '2': 0.35,  # 42% grass
    '3': 0.02, # 1% rocks
    '4': 0.02  # 1% riverstones
}

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("RPG Map with Evolutionary Algorithm")

# Class for evolving maps
class LandscapeGenerator:
    def generate_initial_map(self):
        """Generate a random map."""
        return [[random.choice(TILE_TYPES) for _ in range(MAP_WIDTH)] for _ in range(MAP_HEIGHT)]

    def evaluate_fitness(self, map_data):
        """Evaluate map fitness based on the following criteria:
    
        1. Mountains should be largely connected and placed away from the center.
        2. Rivers should form a ring shape around the center and be connected with a width no larger than 3 tiles.
        3. Grass should be largely connected and placed within the central radius.
        4. Rocks should only appear among grass tiles and be placed within the central radius.
        5. Riverstones should only occur when connecting grass across rivers and be placed in a ring shape around the center.
        6. Most of the landscape should consist of mountains and grass, with a few rivers flowing, and only a decorative amount of rocks and riverstones.
        7. The terrain distribution should closely match the target distribution of tile types.
        
        """
        fitness = 0
        mountain_connectivity = 0
        mountain_placement = 0
        river_placement = 0
        grass_connectivity = 0
        grass_placement = 0
        riverstone_validity = 0
        riverstone_placement = 0
        rock_validity = 0
        rock_placement = 0
        terrain_counts = {tile: 0 for tile in TILE_TYPES}

        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                tile = map_data[y][x]
                terrain_counts[tile] += 1

                # Reward mountain connectivity and placement
                if tile == '0':  # Mountain
                    # Reward mountain connectivity
                    if x > 0 and map_data[y][x - 1] == '0' and \
                        x < MAP_WIDTH - 1 and map_data[y][x + 1] == '0' and \
                        y > 0 and map_data[y - 1][x] == '0' and \
                        y < MAP_HEIGHT - 1 and map_data[y + 1][x] == '0':
                        mountain_connectivity += 5
                    elif x > 0 and map_data[y][x - 1] == '0' or \
                        x < MAP_WIDTH - 1 and map_data[y][x + 1] == '0' or \
                        y > 0 and map_data[y - 1][x] == '0' or \
                        y < MAP_HEIGHT - 1 and map_data[y + 1][x] == '0':
                        mountain_connectivity += 3
					# Penalize mountain disconnection
                    else:
                        mountain_connectivity -= 4

                    distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    distance_ratio = distance_from_center / max_distance
                    # Penalize mountain placement within the radius from the center
                    if distance_from_center <= radius:
                        mountain_placement -= 10 * (1 - distance_ratio)  
                    # Reward mountain placement outside the radius
                    else:
                        mountain_placement += 7 * distance_ratio
                    
                if tile == '1':
                    # Reward river placement in a ring shape
                    ring_radius = radius + 3
                    distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if radius <= distance_from_center <= ring_radius:
                        river_placement += 7  # High reward for river placement in the ring
                    else:
                        river_placement -= 10  # Penalize river placement outside the ring

                # Reward grass connectivity and placement
                if tile == '2':  # Grass
                    if x > 0 and map_data[y][x - 1] == '2' and \
                    	x < MAP_WIDTH - 1 and map_data[y][x + 1] == '2' and \
                        y > 0 and map_data[y - 1][x] == '2' and \
                        y < MAP_HEIGHT - 1 and map_data[y + 1][x] == '2':
                        grass_connectivity += 5
                    elif x > 0 and map_data[y][x - 1] == '2' or \
                    	x < MAP_WIDTH - 1 and map_data[y][x + 1] == '2' or \
                        y > 0 and map_data[y - 1][x] == '2' or \
                        y < MAP_HEIGHT - 1 and map_data[y + 1][x] == '2':
                        grass_connectivity += 3
                    else:
                        grass_connectivity -= 4
                    
                    distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    distance_ratio = distance_from_center / max_distance
                    # Reward grass placement within the radius from the center
                    if distance_from_center <= radius:
                        grass_placement += 7 * distance_ratio 
                    # Penalize grass placement outside the radius
                    else:
                        grass_placement -= 10 * (1 - distance_ratio)

                # Validate rocks
                if tile == '3':  # Rock
                    if (x > 0 and map_data[y][x - 1] == '2') and \
                      (x < MAP_WIDTH - 1 and map_data[y][x + 1] == '2') and \
                      (y > 0 and map_data[y - 1][x] == '2') and \
                      (y < MAP_HEIGHT - 1 and map_data[y + 1][x] == '2'):
                        rock_validity += 5
                    else:
                        rock_validity -= 8
                    
                    # Reward rock placement within the radius from the center
                    distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if distance_from_center <= radius:
                        rock_placement += 7
                    # Penalize grass placement outside the radius
                    else:
                        rock_placement -= 10

                # Validate riverstones
                if tile == '4':  # Riverstone
                    if (x > 0 and map_data[y][x - 1] == '1') and \
                      (x < MAP_WIDTH - 1 and map_data[y][x + 1] == '1') and \
                      (y > 0 and map_data[y - 1][x] == '1') and \
                      (y < MAP_HEIGHT - 1 and map_data[y + 1][x] == '1'):
                        riverstone_validity += 5
                    else:
                        riverstone_validity -= 8
                    
                    # Reward river placement in a ring shape
                    ring_radius = radius + 5
                    distance_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    # High reward for riverstone placement in the ring
                    if radius <= distance_from_center <= ring_radius:
                        riverstone_placement += 7  
                    # Penalize riverstone placement outside the ring
                    else:
                        riverstone_placement -= 10  

        # Calculate the final fitness score
        fitness += river_placement * 3 \
        		+ grass_connectivity \
        		+ grass_placement * 3 \
        		+ mountain_connectivity \
                + mountain_placement * 3 \
                + riverstone_validity * 2 \
                + riverstone_placement * 3 \
                + rock_validity * 2 \
                + rock_placement * 3

        return fitness

    def mutate(self, map_data):
        """Randomly mutate tiles in the map."""
        tile_types = list(TILE_TYPES)
        weights = list(target_distribution.values())

        mutated_map = [row.copy() for row in map_data]
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                if random.random() < MUTATION_RATE:
                    mutated_map[y][x] = random.choices(tile_types, weights)[0]
        return mutated_map

    def crossover(self, parent1, parent2):
        """Perform crossover between two maps."""
        crossover_point = random.randint(0, MAP_WIDTH)
        child = []
        for y in range(MAP_HEIGHT):
            child.append(parent1[y][:crossover_point] + parent2[y][crossover_point:])
        return child

    def evolve(self):
        """Run the evolutionary algorithm to generate the best map."""
        population = [self.generate_initial_map() for _ in range(POPULATION_SIZE)]

        for generation in range(GENERATIONS):
            # Evaluate fitness for the population
            fitness_scores = [self.evaluate_fitness(individual) for individual in population]
            new_population = []

            # Generate new population through selection, crossover, and mutation
            for _ in range(POPULATION_SIZE):
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)

                # Select the best individual among the parents and child
                p1_fitness = self.evaluate_fitness(parent1)
                p2_fitness = self.evaluate_fitness(parent2)
                c_fitness = self.evaluate_fitness(child)

                best = max(p1_fitness, p2_fitness, c_fitness)
                if best == p1_fitness:
                    new_population.append(parent1)
                elif best == p2_fitness:
                    new_population.append(parent2)
                else:
                    new_population.append(child)

            population = new_population
            if generation % 100 == 0:
                print(f"Generation {generation} completed. Best fitness: {max(fitness_scores)}")

        # Return the best map in the final generation
        best_index = fitness_scores.index(max(fitness_scores))
        return population[best_index]

    def tournament_selection(self, population, fitness_scores, k=3):
        """Tournament selection to choose a parent."""
        selected = random.sample(list(enumerate(fitness_scores)), k)
        selected.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness descending
        return population[selected[0][0]]

# RPG Map visualization
class RPGMap:
    def __init__(self, map_data):
        self.map_data = map_data

    def draw(self):
        """Draw the map on the screen."""
        for y, row in enumerate(self.map_data):
            for x, tile in enumerate(row):
                screen.blit(
                    IMAGES[tile],
                    (x * TILE_SIZE, y * TILE_SIZE)
                )

def main():
    clock = pygame.time.Clock()
    running = True

    # Create a folder to save the images if it doesn't exist
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(10):
        print(f"Generating map {i + 1}...")
        # Generate map using evolutionary algorithm
        generator = LandscapeGenerator()
        evolved_map = generator.evolve()
        rpg_map = RPGMap(evolved_map)

        # Save the image to the folder with a unique filename
        image_path = os.path.join(output_folder, f'evolved_map_{i + 1}.png')

        # Draw the map on the screen
        screen.fill((0, 0, 0))  # Clear screen
        rpg_map.draw()  # Draw the map
        pygame.display.flip()

        # Save the screen surface to an image file
        pygame.image.save(screen, image_path)
        print(f"Map {i + 1} saved to {image_path}")

        # Limit frame rate to 30 FPS
        clock.tick(30)

    pygame.quit()

# Run the game
if __name__ == "__main__":
    main()
