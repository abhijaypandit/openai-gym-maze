
#!/usr/bin/env python3


import torch as th
from torch import nn, optim, distributions as dist
import gym
from gym_maze import *
import learn2learn as l2l
from copy import deepcopy
DIM = 5
TIMESTEPS = 1000
TASKS_PER_STEP = 3
from lander_q1 import Agent




def main():
    env1 = gym.make('maze-random-3x3-v0')
    env2 = gym.make('maze-random-3x3-v0')
    env3 = gym.make('maze-random-3x3-v0')
    tasks = [env1,env2,env3]
    num_states=2
    num_actions=4
    ALPHA = 5e-4
    GAMMA = 0.99
    EPSILON = 1.0
    STEPS_PER_TASK = 50000
    BATCH_SIZE = 1024
    meta_agent = Agent(
    num_states=num_states,
    num_actions=num_actions,
    alpha=ALPHA,
    gamma=GAMMA,
    epsilon=EPSILON,
    buffer_size=100000,
    update_freq=1024)
    task_agent = deepcopy(meta_agent)


    maml = l2l.algorithms.MAML(meta_agent.current_network, lr=1e-2)
    opt = optim.Adam(maml.parameters(),lr=ALPHA)


    for i in range(TIMESTEPS):
        step_loss = 0.0
        total_reward = 0.0
        epsiodes=0
        for t in range(TASKS_PER_STEP):
            # Sample a task
            # task_params = task_dist.sample()
            # mu_i, sigma_i = task_params[:DIM], task_params[DIM:]
            env  = tasks[t]

            # Adaptation: Instanciate a copy of model
            task_agent.current_network = maml.clone()

            steps = 0
            terminal=True
            while steps<STEPS_PER_TASK:
            #while True:
                if terminal:
                    state = env.reset()
                    epsiodes+=1
                

                while True:
                    #env.render()
                    action = task_agent.take_action(state)
                    next_state, reward, terminal, _ = env.step(action)
                    experience = (state, action, reward, next_state, terminal)
                    task_agent.update_buffer(experience)
                    step_loss+=task_agent.calculate_loss(int(BATCH_SIZE))
                    

                    total_reward += reward
                    state = next_state
                    steps+=1

                    # Stopping condition for episode - agent encounters terminal state
                    if terminal:
                        break
            # step_loss+=task_agent.calculate_loss(int(STEPS_PER_TASK/2))




        # Meta-learning step: compute gradient through the adaptation step, automatically.
        step_loss = step_loss / TASKS_PER_STEP
        avg_reward = total_reward/(epsiodes)
        print(i, "Average episodic reward",avg_reward,", Step loss:",step_loss.item())
        opt.zero_grad()
        step_loss.backward(retain_graph=True)
        opt.step()


if __name__ == '__main__':
    main()