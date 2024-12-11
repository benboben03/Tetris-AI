import numpy as np
import random
from agent_genetic import GeneticAgent
from src.tetris import Tetris
import cv2

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, elite_fraction):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.population = [GeneticAgent() for _ in range(population_size)]

    def evaluate_fitness(self, agents, render=False, grid_size=(4, 4)):
        fitness_scores = []
        frames = []
        for idx, agent in enumerate(agents):
            env = Tetris()
            total_score = 0
            done = False
            while not done:
                action = agent.get_action(env)
                reward, done = env.step(action, render=False)
                total_score += reward
                if render:
                    frame = env.render(mode='rgb_array')
                    frames.append((idx, frame))
            fitness_scores.append(total_score)
        if render:
            self.render_games(frames, grid_size)
        return fitness_scores

    def render_games(self, frames, grid_size):
        # Organize frames by agent index
        from collections import defaultdict
        agent_frames = defaultdict(list)
        for idx, frame in frames:
            agent_frames[idx].append(frame)

        max_frames = max(len(f) for f in agent_frames.values())
        for frame_idx in range(max_frames):
            grid_frames = []
            for idx in range(self.population_size):
                if frame_idx < len(agent_frames[idx]):
                    frame = agent_frames[idx][frame_idx]
                else:
                    # If agent finished, show last frame
                    frame = agent_frames[idx][-1]
                grid_frames.append(frame)

            # Create grid image
            grid_img = self.create_grid(grid_frames, grid_size)
            cv2.imshow("Genetic Algorithm Tetris", grid_img)
            key = cv2.waitKey(1)
            if key == 27:  # Esc key to stop
                cv2.destroyAllWindows()
                return

    def create_grid(self, frames, grid_size):
        rows = []
        idx = 0
        frame_height, frame_width, _ = frames[0].shape
        for _ in range(grid_size[0]):
            row_frames = []
            for _ in range(grid_size[1]):
                if idx < len(frames):
                    frame = frames[idx]
                else:
                    # Fill with black images if not enough frames
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                row_frames.append(frame)
                idx += 1
            row = np.hstack(row_frames)
            rows.append(row)
        grid_img = np.vstack(rows)
        return grid_img

    def select_parents(self, fitness_scores):
        # Normalize fitness scores
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            selection_probs = [1 / len(fitness_scores)] * len(fitness_scores)
        else:
            selection_probs = [f / total_fitness for f in fitness_scores]
        parents = np.random.choice(self.population, size=2, p=selection_probs)
        return parents

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            # Single-point crossover
            crossover_point = random.randint(1, len(parent1.weights) - 1)
            child_weights = np.concatenate(
                (parent1.weights[:crossover_point], parent2.weights[crossover_point:])
            )
            return GeneticAgent(weights=child_weights)
        else:
            return GeneticAgent(weights=parent1.weights.copy())

    def mutate(self, agent):
        for i in range(len(agent.weights)):
            if random.random() < self.mutation_rate:
                # Mutate weight with Gaussian noise
                agent.weights[i] += np.random.normal()
        return agent

    def evolve(self, render=False, grid_size=(4, 4)):
        # Evaluate fitness
        fitness_scores = self.evaluate_fitness(self.population, render=render, grid_size=grid_size)
        new_population = []

        # Elitism: keep the top-performing agents
        elite_size = int(self.elite_fraction * self.population_size)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        elite_agents = [self.population[i] for i in elite_indices]
        new_population.extend(elite_agents)

        # Generate the rest of the new population
        while len(new_population) < self.population_size:
            parent1, parent2 = self.select_parents(fitness_scores)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        max_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        min_fitness = min(fitness_scores)  # Add this line to calculate min fitness
        return max_fitness, avg_fitness, min_fitness  # Return all three values
