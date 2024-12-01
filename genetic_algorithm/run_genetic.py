from genetic_algorithm import GeneticAlgorithm
from agent_genetic import GeneticAgent
from tetris import Tetris
import numpy as np

def train_genetic_algorithm():
    generations = 50
    population_size = 16  # Adjusted for grid display (e.g., 4x4 grid)
    mutation_rate = 0.1
    crossover_rate = 0.7
    elite_fraction = 0.2  # Keep top 20% agents
    ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, elite_fraction)

    best_fitness = -float('inf')
    best_agent = None

    for generation in range(generations):
        print(f"Generation {generation}")
        # Set render=True for visualization
        max_fitness, avg_fitness = ga.evolve(render=True, grid_size=(4, 4))
        print(f"Max Fitness = {max_fitness}, Avg Fitness = {avg_fitness}")

        # Keep track of the best agent
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            # The best agent is the first in the sorted elite agents
            best_agent = ga.population[np.argmax([ga.evaluate_fitness([agent])[0] for agent in ga.population])]

    print("Best Weights:", best_agent.weights)
    return best_agent

def test_agent(agent):
    env = Tetris()
    done = False
    total_score = 0
    while not done:
        action = agent.get_action(env)
        reward, done = env.step(action, render=True)
        total_score += reward
    print(f"Final Score: {env.score}, Total Reward: {total_score}")

if __name__ == '__main__':
    best_agent = train_genetic_algorithm()
    test_agent(best_agent)
