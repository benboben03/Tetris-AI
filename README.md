# CS 175 Tetris AI

### genetic_algorithm Directory:
| File Name          | Functionality                                                                                                                                         |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `agent_genetic.py` | Implements the `GeneticAgent` class, which is used to represent the individual agents in the population. All these agents use a weight vector to decide the game states and select actions. |
| `genetic_algorithm.py` | Contains the `GeneticAlgorithm` class, which handles the evolution process of the agent population. This includes crossover, mutation, fitness evaluation, and elitism. `crossover(parent1, parent2)` combines weights of two parents to create offspring. `mutate(agent)` applies Gaussian noise for random exploration. |
| `run_genetic.py`   | This is the main training loop of the genetic algorithm. It tracks fitness trends of successive generations by plotting them on a line graph and saves the best performing agent using pickle. |
| `run_best_agent.py`| Loads and tests the best-performing agent saved after training. Visualizes the game in real-time and saves the gameplay as an MP4 video.                                                   |

## Disclaimer
All models uses tetris.py implementation from https://github.com/vietnh1009/Tetris-deep-Q-learning-pytorch/tree/master
All credit to the creator. Please note that some of our models uses a modified tetris.py which is not in the repository, to avoid confusion.
