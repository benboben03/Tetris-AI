"""This module contains the mapping for the rewards that the agent can receive."""
from dataclasses import dataclass


# @dataclass
# class RewardsMapping:
#     """Mapping for the rewards that the agent can receive.

#     The mapping can be extended to include additional rewards.
#     """

#     alife: float = 1
#     clear_line: float = 1
#     game_over: float = 0
#     invalid_action: float = -0.1

@dataclass
class RewardsMapping:
    """Mapping for the rewards that the agent can receive."""

    alife: float = 1
    clear_line: float = 1
    game_over: float = 0
    invalid_action: float = -0.1
    hole_penalty: float = 0.1
    height_penalty: float = 0.1
    piece_placed: float = 0.5
    step_penalty: float = -0.01  # Negative reward for each step to encourage faster play
