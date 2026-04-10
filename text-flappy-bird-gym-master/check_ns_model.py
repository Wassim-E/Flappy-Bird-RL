import os
import sys
import gymnasium as gym
import time
import pickle
import numpy as np
import text_flappy_bird_gym


def load_q_policy(path):
    with open(path, 'rb') as f:
        # If your policy was saved as a dict of Q-tables
        q_table = pickle.load(f)
    return q_table 

if __name__ == '__main__':
    # Adjust path as needed
    try:
        q_table = load_q_policy('weights/BEST_mcns_policy.pkl')
    except FileNotFoundError:
        print("Warning: Policy file not found. Running with random actions.")
        q_table = {}

    env = gym.make('TextFlappyBird-v0', height=15, width=20, pipe_gap=4)
    state, _ = env.reset()

    while True:
        # Rendering
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Parsed State (h_dist, v_dist): {state}")
        print(env.render())
        time.sleep(0.1)

        # Action Selection
        if state is not None and state in q_table:
            # Simple argmax for the 2 actions (0: stay, 1: flap)
            action = int(np.argmax(q_table[state]))
        else:
            action = env.action_space.sample()

        state, reward, done, _, info = env.step(action)
        
        if done:
            print("Game Over!")
            break
    env.close()