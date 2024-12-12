import os
import gc
import pickle
from genetic_algorithm import GeneticAlgorithm
from agent_genetic import GeneticAgent
from genetic_algorithm.tetris import Tetris
import numpy as np
import matplotlib.pyplot as plt


def train_genetic_algorithm():
    generations = 30
    population_size = 16  # 16 agents per generation
    mutation_rate = 0.1   # 10% chance of agent mutated
    crossover_rate = 0.7  # 70% of pairings through crossover  
    elite_fraction = 0.2  # Keep top 20% agents = 3 top agents + 13 children
    ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, elite_fraction)

    best_fitness = -float('inf')
    best_agent = None

    # To track fitness trends
    fitness_trends = {
        "max": [],
        "avg": [],
        "min": []
    }

    for generation in range(generations):
        generation += 1
        print(f"Generation {generation}")
        # Evolve population and get fitness statistics
        max_fitness, avg_fitness, min_fitness = ga.evolve(render=True, grid_size=(4, 3))  # Render games
        print(f"Max Fitness = {max_fitness}, Avg Fitness = {avg_fitness}, Min Fitness = {min_fitness}")
        gc.collect()  # clear unused memory

        # Append fitness trends
        fitness_trends["max"].append(max_fitness)
        fitness_trends["avg"].append(avg_fitness)
        fitness_trends["min"].append(min_fitness)

        # Keep track of the best agent
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_agent = ga.population[np.argmax([ga.evaluate_fitness([agent])[0] for agent in ga.population])]

            # Save the best model
            save_best_agent(best_agent, "best_agent.pkl")
            print(f"Best model saved with fitness {best_fitness}")

        # Save fitness trend plot
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
        plot_file = os.path.join(script_dir, "fitness_trends.png")
        plot_fitness_trends(fitness_trends, file_name=plot_file)

    print("Best Weights:", best_agent.weights)
    return best_agent


def plot_fitness_trends(fitness_trends, file_name="fitness_trends.png"):
    """Plots and saves the max, avg, and min fitness trends across generations."""
    generations = range(len(fitness_trends["max"]))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_trends["max"], label="Max Fitness", linewidth=2)
    plt.plot(generations, fitness_trends["avg"], label="Avg Fitness", linestyle="--")
    plt.plot(generations, fitness_trends["min"], label="Min Fitness", linestyle=":")

    plt.title("Fitness Trends Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)

    plt.savefig(file_name)
    print(f"Fitness trends plot saved as {file_name}")
    plt.close()


def save_best_agent(agent, file_name="best_agent.pkl"):
    """Saves the best agent to a file in the script's directory using pickle."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, file_name)

    with open(full_path, "wb") as f:
        pickle.dump(agent, f)
    print(f"Best agent saved to {full_path}")

def test_agent(agent):
    env = Tetris()
    done = False
    total_score = 0
    while not done:
        action = agent.get_action(env)
        reward, done = env.step(action, render=True)  # Render the game
        total_score += reward
    print(f"Final Score: {env.score}, Total Reward: {total_score}")


if __name__ == '__main__':
    best_agent = train_genetic_algorithm()
    test_agent(best_agent)
