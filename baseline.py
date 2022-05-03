import sys
import numpy as np
import math
import random

import gym
import gym_maze
from gym_maze import *

from stable_baselines3 import DQN

# Create environment
env = gym.make('maze-random-5x5-v0')

# Instantiate the agent
model = DQN('MlpPolicy', env, verbose=1)
# Train the agent
model.learn(total_timesteps=50000)
# Save the agent
#model.save("maze")
#del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
#model = DQN.load("maze", env=env)

# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

#del model # remove to demonstrate saving and loading

model = DQN.load("maze")

obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #print("DEBUG: action = ", action)
    env.render()
    if done:
      obs = env.reset()