import pickle
import os
import cv2
from src.tetris import Tetris


def load_best_agent(file_name="best_agent.pkl"):
    """Loads the best agent from a file using pickle."""
    absolute_path = os.path.abspath(file_name)
    directory = os.path.dirname(absolute_path)
    print(f"Loading best agent from: {absolute_path}")

    file_size = os.path.getsize(file_name)
    print(f"Best agent saved to {absolute_path} ({file_size / (1024 * 1024):.2f} MB)")
    with open(file_name, "rb") as f:
        return pickle.load(f), directory


def test_agent(agent, video_filename):
    """Run the best agent and record the gameplay."""
    env = Tetris()
    done = False
    total_score = 0

    print("Starting game with the best agent...")

    frame_width = env.width * env.block_size * 2  # Multiply by 2 to account for extra board
    frame_height = env.height * env.block_size
    fps = 30  # Frames per second
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    while not done:
        # Get the best agent's action
        action = agent.get_action(env)
        # Perform the action in the Tetris environment and render
        reward, done = env.step(action, render=False)
        total_score += reward
        frame = env.render(mode='rgb_array')
        out.write(frame)

    print(f"Final Score: {env.score}, Total Reward: {total_score}")

    out.release()
    print(f"Gameplay video saved as {video_filename}")


if __name__ == "__main__":
    # Load the best agent and get its directory
    best_agent, agent_directory = load_best_agent("genetic_algorithm/best_agent.pkl")
    video_filename = os.path.join(agent_directory, "tetris_gameplay.mp4")

    # Test and record the agent
    test_agent(best_agent, video_filename=video_filename)
