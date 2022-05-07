import random
import copy
import numpy as np
import gym
from deepqlearn import Agent
import gym_maze

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import cherry as ch
from cherry import envs


ACTION_DISCRETISATION = 4
DISCOUNT = 0.99
EPSILON = 0.8
HIDDEN_SIZE = 128
LEARNING_RATE = 0.01
MAX_STEPS = 100000
BATCH_SIZE = 32
REPLAY_SIZE = 100000
TARGET_UPDATE_INTERVAL = 512
UPDATE_INTERVAL = 1
UPDATE_START = 1000
SEED = 42
EVAL_NUM_EPISODES=10
EPSILON_UPDATE_INTERVAL=100

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class DQN(nn.Module):

    def __init__(self, hidden_size, num_actions=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, hidden_size),
                  nn.Tanh(),
                  nn.Linear(hidden_size, num_actions)]
        self.dqn = nn.Sequential(*layers)
        self.egreedy = ch.nn.EpsilonGreedy(EPSILON)

    def forward(self, state):
        values = self.dqn(state)
        action = self.egreedy(values)
        return action, values


def create_target_network(network):
    target_network = copy.deepcopy(network)
    for param in target_network.parameters():
        param.requires_grad = False
    return target_network


def convert_discrete_to_continuous_action(action):
    return action.to(dtype=torch.float32) - ACTION_DISCRETISATION // 2



def evaluate_agent(env,agent,max_eps,max_steps_per_eps,f):
    eps=0
    total_reward=0.0
    
    while eps<max_eps:
        eps+=1
        steps=0
        while steps<max_steps_per_eps:
            steps+=1
            state=env.reset()
            # action = agent(state)[1].max(dim=1, keepdim=True)[0]
            exp = env.run(f,steps=1)
            total_reward+=exp.reward()[-1]
            terminal = exp.done()[-1]
            if terminal==1.0:
                break

    return total_reward/max_eps #Avg episodic reward


def main(env):
    env = gym.make(env)
    env.seed(SEED)
    env = envs.Torch(env)
    env = envs.ActionLambda(env, convert_discrete_to_continuous_action)
    env = envs.Logger(env)
    env = envs.Runner(env)

    replay = ch.ExperienceReplay()
    agent = DQN(HIDDEN_SIZE, ACTION_DISCRETISATION)
    target_agent = create_target_network(agent)
    optimiser = optim.Adam(agent.parameters(), lr=LEARNING_RATE)

    def get_random_action(state):
        action = torch.tensor([[random.randint(0, ACTION_DISCRETISATION - 1)]])
        return action

    def get_action(state):
        # Original sampling (for unit test)
        if random.random() < EPSILON:
         action = torch.tensor([[random.randint(0, ACTION_DISCRETISATION - 1)]])
        else:
         action = agent(state)[1].argmax(dim=1, keepdim=True)
        return action
        # return agent(state)[0]
    
    def take_action(state):
        global EPSILON, EPSILON_UPDATE_INTERVAL
        if step % EPSILON_UPDATE_INTERVAL == 0:
            EPSILON = max(0.9999*EPSILON, 0.01) # decay epsilon

        max_a = action = agent(state)[1].argmax(dim=1, keepdim=True)

        rand_action = env.action_space.sample()

        action = np.random.choice([max_a,rand_action],1,p=[1.0-EPSILON,EPSILON])


        #action = env.action_space.sample()
        return torch.tensor(action[0])

    def get__greedy_action(state):

        return agent(state)[1].max(dim=1, keepdim=True)[0]

    for step in range(1, MAX_STEPS + 1):
        with torch.no_grad():
            if step < UPDATE_START:
                replay += env.run(get_random_action, steps=1)
            else:
                replay += env.run(take_action, steps=1)

            replay = replay[-REPLAY_SIZE:]

        if step > UPDATE_START and step % UPDATE_INTERVAL == 0:
            # Randomly sample a batch of experience
            batch = random.sample(replay, BATCH_SIZE)
            batch = ch.ExperienceReplay(batch)

            # Compute targets
            with torch.no_grad():
                target_values = target_agent(batch.next_state())[1].max(dim=1, keepdim=True)[0]
                target_values = batch.reward() + DISCOUNT * (1 - batch.done()) * target_values

            # Update Q-function by one step of gradient descent
            pred_values = agent(batch.state())[1].gather(1, batch.action())
            value_loss = F.mse_loss(pred_values, target_values)
            optimiser.zero_grad()
            value_loss.backward()
            optimiser.step()

        if step > UPDATE_START and step % TARGET_UPDATE_INTERVAL == 0:
            # Update target network
            target_agent = create_target_network(agent)
        
        # print("Iteration :{}, Eval score: {} ".format(step,evaluate_agent(env,agent,EVAL_NUM_EPISODES,MAX_STEPS,get__greedy_action)))


if __name__ == '__main__':
    main('maze-sample-3x3-v0')