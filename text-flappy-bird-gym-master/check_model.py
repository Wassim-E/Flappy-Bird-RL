import os
import sys
import gymnasium as gym
import time
import pickle
import numpy as np

import text_flappy_bird_gym

def obs_to_state(obs):
	return tuple(obs.flatten())

def load_policy(path):
	with open(path, 'rb') as f:
		data = pickle.load(f)
	Q = data['Q']
	policy = {}
	for state in Q:
		best_action = int(np.argmax(Q[state]))
		policy[state] = best_action
	return policy

if __name__ == '__main__':
	# Load the trained policy
	policy = load_policy('screen_mc_agent.pkl')

	# Create the environment
	env = gym.make('TextFlappyBird-screen-v0', height=15, width=20, pipe_gap=4)
	obs, _ = env.reset()

	while True:
		state = obs_to_state(obs)
		action = policy.get(state, env.action_space.sample())
		obs, reward, done, _, info = env.step(action)
		os.system("cls")
		sys.stdout.write(env.render())
		time.sleep(0.2)
		if done:
			break
	env.close()
