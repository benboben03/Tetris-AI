import numpy as np
import torch

class GeneticAgent:
    def __init__(self, weights=None):
        # Initialize weights for the features:
        # lines_cleared, holes, bumpiness, height
        if weights is None:
            # Random weights between -1 and 1
            self.weights = np.random.uniform(-1, 1, 4)
        else:
            self.weights = weights

    def get_action(self, env):
        # Get possible next states
        next_states = env.get_next_states()
        best_value = -float('inf')
        best_action = None

        for action, features in next_states.items():
            # Compute value using linear combination of features and weights
            value = np.dot(self.weights, features.numpy())
            if value > best_value:
                best_value = value
                best_action = action

        return best_action
