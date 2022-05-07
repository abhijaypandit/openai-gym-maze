
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

from gym_maze import *
import math

from config import RENDER_MAZE

class Network(Module):

    def __init__(self, input_shape, output_shape):
        Module.__init__(self)
        self.fc1 = Linear(input_shape, 8)
        self.fc2 = Linear(8, 16)
        self.fc3 = Linear(16, output_shape)

    def forward(self, inputs):
        outputs = relu(self.fc1(inputs))
        outputs = relu(self.fc2(outputs))
        outputs = self.fc3(outputs)

        return outputs

class Agent:

    def __init__(self, env, num_states, num_actions, alpha, gamma, epsilon, buffer_size, update_freq,eps_upd_freq):
        self.env = env

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_freq=eps_upd_freq

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
        if self.step_count % self.eps_freq == 0:
            self.epsilon = max(0.8*self.epsilon, 0.01) # decay epsilon
        #self.epsilon = max(0.001, min(0.8, 1.0 - math.log10((self.step_count+1)/
        #                np.prod(tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int)), dtype=float) / 10.0)))
        with torch.no_grad():
            max_a = int(np.argmax(self.current_network(torch.from_numpy(state).float()).numpy()))
        rand_action = self.env.action_space.sample()
        if self.epsilon>0.05:
            action = np.random.choice([max_a,rand_action],1,p=[1.0-self.epsilon,self.epsilon])
        else:
            action = max_a
        #action = env.action_space.sample()
        return action

    # Function to add experience to replay buffer
    def update_buffer(self, experience):
        self.replay_buffer.append(experience)

    # Function to sample a mini-batch from replay buffer
    def sample_buffer(self, batch_size):
        batch =  random.sample(self.replay_buffer, batch_size)

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
            terminals = np.array([1 if x==True else 0 for x in terminals])

            with torch.no_grad():
                target_q = self.target_network(torch.tensor(next_states).float())
            current_q = self.current_network(torch.tensor(current_states).float())

            target = torch.tensor(rewards) + torch.max(target_q, axis=1)[0] * torch.from_numpy(np.ones(batch_size) - terminals)
            current = current_q.gather(dim=1, index=torch.from_numpy(actions.reshape(actions.shape[0], -1)))

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
        loss = 0
        
        if len(self.replay_buffer)>batch_size:
            current_states, actions, rewards, next_states, terminals = self.sample_buffer(batch_size)

            with torch.no_grad():
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
                #self.target_network = deepcopy(self.current_network) # update target network
                states = self.current_network.state_dict()
                states = {k.replace('module.',''): v for k, v in states.items()}
                self.target_network.load_state_dict(states)

        return loss

if __name__ == '__main__':
    env = gym.make("maze-sample-3x3-v0")

    num_actions = 4 # up, down, left, right
    num_states = 2 # (x,y) coordinates

    ALPHA = 5e-4
    GAMMA = 0.9
    EPSILON = 1.0

    agent = Agent(
        env = env,
        num_states=num_states,
        num_actions=num_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        buffer_size=1000,
        update_freq=1028,
        eps_upd_freq=5000
    )
    
    #summary(agent.current_network, input_size=(num_states,))
    #summary(agent.target_network, input_size=(num_states,))    

    EPISODES = 20000
    STEPS = 50
    BATCH_SIZE = 128

    raw_rewards = []
    avg_rewards = []

    plt.figure("Training Performance")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()

    for _ in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for _ in range(STEPS):
            if RENDER_MAZE:
                env.render()
            action = agent.take_action(state)
            next_state, reward, terminal, _ = env.step(action)
            experience = (state, action, reward, next_state, terminal)
            agent.update_buffer(experience)
            agent.train(batch_size=BATCH_SIZE)

            total_reward += reward
            state = next_state

            # Stopping condition for episode - agent encounters terminal state
            if terminal:
                break
        
        raw_rewards.append(total_reward) # actual episodic reward
        avg_rewards.append(np.average(raw_rewards[-50:])) # running average of last 'n' rewards


        #print("\rEpisode {}: raw_reward = {:.4f}, avg_reward = {:.4f}".format(len(raw_rewards), raw_rewards[-1], avg_rewards[-1]), end="\x1b[2K")
        if len(raw_rewards)%1==0:
            print("\rEpisode {}: raw_reward = {:.4f}, avg_reward = {:.4f}, epsilon = {:.4f}".format(len(raw_rewards), raw_rewards[-1], avg_rewards[-1], agent.epsilon))
    
        # plt.plot(raw_rewards, color='blue', label="raw")
        # plt.plot(avg_rewards, color='red', label="average")
        # plt.pause(0.05)
        # plt.draw()
    
    # plt.show()

    # plt.plot(raw_rewards, label="raw")
    # plt.plot(avg_rewards, label="average")
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.legend(loc="lower right")
    # plt.grid()
    # plt.savefig('rewards.png')
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