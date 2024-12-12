# CS 175 Tetris AI
### Tetris Environment:
`tetris.py:`
- Adapted from user made Tetris environment as seen in DISCLAIMER.
- The original environment’s reward system was changed to reward lines clear linearly rather than quadratically. The state space was also experimented with and made more complex. 
- The environment was adapted for GA to be able to handle the population based approach.
### dqn Directory:
All code in `dqn.ipynb`
| Class/Function          | Functionality                                                                                                                                         |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DQN` | Torch neural networ-k consisting of 3 sequential layers, utilizing ReLU activation and batch normalization. The output is the selected action index to select. Xavier initialization is applied to the weights. There are two implementations of DQN, one with size 64 layers for the environment with 4 features, and one  with size 128 layers for the environment with 8 features. |
| `train` | Runs game simulations, choosing either the best known action or a random action, based on the epsilon-greedy algorithm. State-action pairs and their reward are stored in an experience replay before. After 30,000 experiences are stored, the model begins fitting to sampled experiences. Loop runs for 3,000 games (epochs) but can be terminated early. |
| `record_game` | Records a single game that plays using the trained model, utilizing opencv |library
| `evaluate_model` | Runs num_games games and returns the average score, average tetrominoes survived, and average lines cleared. |


### genetic_algorithm Directory:
| File Name          | Functionality                                                                                                                                         |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `agent_genetic.py` | Implements the `GeneticAgent` class, which is used to represent the individual agents in the population. All these agents use a weight vector to decide the game states and select actions. |
| `genetic_algorithm.py` | Contains the `GeneticAlgorithm` class, which handles the evolution process of the agent population. This includes crossover, mutation, fitness evaluation, and elitism. `crossover(parent1, parent2)` combines weights of two parents to create offspring. `mutate(agent)` applies Gaussian noise for random exploration. |
| `run_genetic.py`   | This is the main training loop of the genetic algorithm. It tracks fitness trends of successive generations by plotting them on a line graph and saves the best performing agent using pickle. |
| `run_best_agent.py`| Loads and tests the best-performing agent saved after training. Visualizes the game in real-time and saves the gameplay as an MP4 video.                                                   |

### ppo Directory
Attempt to train PPO model onto tetris is located in `ppo.ipynb`. However this was a failed attempt due to lack of time and resources. However this is good starter code for us to be able to develop onto more in the future.

## Disclaimer
All models uses tetris.py implementation from https://github.com/vietnh1009/Tetris-deep-Q-learning-pytorch/tree/master
All credit to the creator. Please note that some of our models uses a modified tetris.py which is not in the repository, to avoid confusion.
