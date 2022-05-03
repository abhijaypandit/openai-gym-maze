import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import Module, Linear, MSELoss
from torch.nn.functional import relu
from torch.optim import Adam
# from torchsummary import summary

from collections import deque
from copy import deepcopy
import random

env = gym.make("LunarLander-v2")

"""
### Action Space
There are four discrete actions available: do nothing, fire left
orientation engine, fire main engine, fire right orientation engine.

### Observation Space
There are 8 states: the coordinates of the lander in `x` & `y`, its linear
velocities in `x` & `y`, its angle, its angular velocity, and two bool
"""

num_actions = 4
num_states = 8

class Network(Module):

    def __init__(self, input_shape, output_shape):
        Module.__init__(self)
        self.fc1 = Linear(input_shape, 32)
        self.fc2 = Linear(32, 64)
        self.fc3 = Linear(64, output_shape)

    def forward(self, inputs):
        outputs = relu(self.fc1(inputs))
        outputs = relu(self.fc2(outputs))
        outputs = self.fc3(outputs)

        return outputs

class Agent:

    def __init__(self, num_states, num_actions, alpha, gamma, epsilon, buffer_size, update_freq):
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Replay buffer
        self.replay_buffer = deque(maxlen=buffer_size)

        # Current and target networks
        self.current_network = Network(num_states, num_actions)
        self.target_network = deepcopy(self.current_network)
        self.update_freq = update_freq
        self.step_count = 0

        # Loss and optimizer
        self.loss = MSELoss()
        self.optimizer = Adam(self.current_network.parameters(), lr=alpha)

    # Function to select action using epsilon-greedy
    def take_action(self, state):
        if self.step_count % self.update_freq == 0:
            self.epsilon = max(0.9*self.epsilon, 0.01) # decay epsilon

        if np.random.rand() > self.epsilon:
            self.current_network.eval()
            with torch.no_grad():
                action = np.argmax(self.current_network(torch.from_numpy(state).type(torch.float)).numpy())
        else:
            action = env.action_space.sample()

        return action

    # Function to add experience to replay buffer
    def update_buffer(self, experience):
        self.replay_buffer.append(experience)

    # Function to sample a mini-batch from replay buffer
    def sample_buffer(self, batch_size):
        batch =  random.sample(self.replay_buffer, int(batch_size))

        current_states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        
        for sample in batch:
            current_states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_states.append(sample[3])
            terminals.append(sample[4])

        # batch -> {current_state, action, reward, next_state, terminal}
        return (np.array(current_states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminals))

    # Function to train agent
    def train(self, batch_size):
        self.current_network.train()
        self.step_count += 1 # incerement step counter

        if self.step_count >= batch_size:
            current_states, actions, rewards, next_states, terminals = self.sample_buffer(batch_size)

            target_q = self.target_network(torch.tensor(next_states))
            current_q = self.current_network(torch.tensor(current_states))

            target = torch.tensor(rewards) + torch.max(target_q, axis=1)[0] * torch.tensor(np.ones(batch_size) - terminals)
            current = current_q.gather(dim=1, index=torch.tensor(actions.reshape(actions.shape[0], -1)))

            # Calculate loss
            loss = self.loss(target.unsqueeze(1).double(), current.double())

            # Backpropagate loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.step_count % self.update_freq == 0:
                self.target_network = deepcopy(self.current_network) # update target network

    def calculate_loss(self, batch_size):
        self.current_network.train()
        self.step_count += 1 # incerement step counter

        if len(self.replay_buffer)>batch_size:
            current_states, actions, rewards, next_states, terminals = self.sample_buffer(batch_size)

            target_q = self.target_network(torch.tensor(next_states).float())
            current_q = self.current_network(torch.tensor(current_states).float())

            target = torch.tensor(rewards) + torch.max(target_q, axis=1)[0] * torch.tensor(np.ones(batch_size) - terminals)
            current = current_q.gather(dim=1, index=torch.tensor(actions.reshape(actions.shape[0], -1)))

            # Calculate loss
            loss = self.loss(target.unsqueeze(1).double(), current.double())

            # Backpropagate loss
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            if self.step_count % self.update_freq == 0:
                self.target_network = deepcopy(self.current_network) # update target network

        return loss

# ALPHA = 0.001
# GAMMA = 0.99
# EPSILON = 1

# agent = Agent(
#     num_states=num_states,
#     num_actions=num_actions,
#     alpha=ALPHA,
#     gamma=GAMMA,
#     epsilon=EPSILON,
#     buffer_size=100000,
#     update_freq=1024
# )

# #summary(agent.current_network, input_size=(num_states,))
# #summary(agent.target_network, input_size=(num_states,))

# EPISODES = 2000
# BATCH_SIZE = 32

# raw_rewards = []
# avg_rewards = []

# for _ in range(EPISODES):
# #while True:
#     state = env.reset()
#     total_reward = 0

#     while True:
#         #env.render()
#         action = agent.take_action(state)
#         next_state, reward, terminal, _ = env.step(action)
#         experience = (state, action, reward, next_state, terminal)
#         agent.update_buffer(experience)
#         agent.train(batch_size=BATCH_SIZE)

#         total_reward += reward
#         state = next_state

#         # Stopping condition for episode - agent encounters terminal state
#         if terminal:
#             break
    
#     raw_rewards.append(total_reward) # actual episodic reward
#     avg_rewards.append(np.average(raw_rewards[-30:])) # running average of last 'n' rewards

#     # Stopping condition for training - running average of rewards >= 200
#     if avg_rewards[-1] >= 200:
#         torch.save(agent.current_network.state_dict(), "lander_q1.pth")
#         print("Agent trained!")
#         break

#     print("\rEpisode {}: raw_reward = {:.4f}, avg_reward = {:.4f}".format(len(raw_rewards), raw_rewards[-1], avg_rewards[-1]), end="\x1b[2K")

# plt.plot(raw_rewards, label="raw")
# plt.plot(avg_rewards, label="average")
# plt.title("Episodic Reward")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.legend(loc="lower right")
# plt.grid()
# plt.savefig('lander_q1.png')
# '''
# agent.current_network.load_state_dict(torch.load('lander_q1.pth'))

# for _ in range(5):
#     state = env.reset()
#     total_reward = 0

#     while True:
#         env.render()
#         action = agent.take_action(state)
#         next_state, reward, terminal, _ = env.step(action)
#         state = next_state

#         if terminal:
#             break
# '''