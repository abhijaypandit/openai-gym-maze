from sqlite3 import adapt
import torch as th
from torch import nn, optim, distributions as dist
import gym
from gym_maze import *
import learn2learn as l2l
from copy import deepcopy
from deepqlearn import Agent


th.autograd.set_detect_anomaly(True)

DIM = 5
TIMESTEPS = 500
TASKS_PER_STEP = 3

def main():
    env1 = gym.make('maze-random-3x3-v0')
    env2 = gym.make('maze-random-3x3-v0')
    env3 = gym.make('maze-random-3x3-v0')
    tasks = [env1, env2, env3]

    num_states=2
    num_actions=4
    
    ALPHA = 5e-4
    GAMMA = 0.99
    EPSILON = 1.0
    STEPS_PER_TASK = 1024
    BATCH_SIZE = 32
    
    meta_agent = Agent(
        env = env1,
        num_states=num_states,
        num_actions=num_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        buffer_size=10000,
        update_freq=512
    )
    #task_agent = deepcopy(meta_agent)
    task_agent1 = Agent(
        env = env1,
        num_states=num_states,
        num_actions=num_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        buffer_size=10000,
        update_freq=512
    )

    task_agent2 = Agent(
        env = env1,
        num_states=num_states,
        num_actions=num_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        buffer_size=10000,
        update_freq=512
    )

    task_agent3 = Agent(
        env = env1,
        num_states=num_states,
        num_actions=num_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        buffer_size=10000,
        update_freq=512
    )
    maml = l2l.algorithms.MAML(meta_agent.current_network, lr=5e-4)
    opt = optim.Adam(maml.parameters(),lr=ALPHA)

    agents = [task_agent1,task_agent2,task_agent3]
    for i in range(TIMESTEPS):
        step_loss = 0.0
        total_reward = 0.0
        epsiodes = 0
    
        for t in range(len(tasks)):
            env  = tasks[t]
            task_agent = agents[t]
            # Adaptation: Instantiate a copy of model
            task_agent.current_network = maml.clone()

            steps = 0
            while steps<STEPS_PER_TASK:
            #while True:
                epsiodes+=1
                state = env.reset()

                while True:
                    #env.render()
                    action = task_agent.take_action(state)
                    next_state, reward, terminal, _ = env.step(action)
                    experience = (state, action, reward, next_state, terminal)
                    #print("experience: ", experience)
                    task_agent.update_buffer(experience)
                    #print("cal_loss = ", task_agent.calculate_loss(int(BATCH_SIZE)))
                    step_loss += task_agent.calculate_loss(int(BATCH_SIZE))

                    total_reward += reward
                    state = next_state
                    steps+=1

                    # Stopping condition for episode - agent encounters terminal state
                    if terminal or steps == STEPS_PER_TASK:
                        break

            # step_loss+=task_agent.calculate_loss(int(STEPS_PER_TASK/2))
            #print("Step {} :: Task {} :: Loss {}".format(i,t,step_loss))

            # Take a gradient step on the loss and updates the cloned parameters in place
            # task_agent.current_network.adapt(step_loss)
            
            # Adaptation: Evaluate the effectiveness of adaptation
            # adapt_loss = task_agent.calculate_loss(int(BATCH_SIZE))

            # Accumulate the error over all tasks
            # step_loss += adapt_loss


        # Meta-learning step: compute gradient through the adaptation step, automatically.
        step_loss = step_loss / (TASKS_PER_STEP*STEPS_PER_TASK)
        avg_reward = total_reward/(epsiodes)
        print(i, "Average episodic reward",avg_reward,", Step loss:",step_loss.item())
        opt.zero_grad()
        #step_loss.backward(retain_graph=True)
        step_loss.backward()
        opt.step()
        th.save(meta_agent.current_network.state_dict(),'./meta_multi_agent_3x3.pth')


if __name__ == '__main__':
    main()


