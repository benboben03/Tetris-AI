import cv2
import gymnasium as gym

from tetris_gymnasium.envs.tetris import Tetris

if __name__ == "__main__":
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    env.reset(seed=42)
    print("_____OBSERVATION SPACE_____ \n")
    print("Observation Space", env.observation_space)
    print("Sample observation", env.observation_space.sample()) # Get a random observation

    print("_____ACTION SPACE_____ \n")
    print("Action Space", env.action_space)
    print("Sample observation", env.action_space.sample()) # Get a random observation
    
    terminated = False
    while not terminated:
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        key = cv2.waitKey(100) # timeout to see the movement
    print("Game Over!")

