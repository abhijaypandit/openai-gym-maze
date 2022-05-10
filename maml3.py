from copy import deepcopy
import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import random
import time
from agent import Agent
import learn2learn as l2l

from gym_maze import *
torch.autograd.set_detect_anomaly(True)

DIM = 5
TIMESTEPS = 40000
TASKS_PER_STEP = 1

def main():
    env1 = gym.make('maze-random-3x3-v0')
    env2 = gym.make('maze-random-3x3-v0')
    env3 = gym.make('maze-random-3x3-v0')
    tasks = [env1, env2, env3]
    # tasks = [env1]

    num_states=2
    num_actions=4
    
    ALPHA = 5e-4
    GAMMA = 0.99
    EPSILON = 1.0
    STEPS_PER_TASK = 800
    EPISODES_PER_TASK=1
    LR_META=5e-4
    # BATCH_SIZE = 32
    
    meta_agent =Agent(state_size=2, action_size=4, seed=0)
    maml = l2l.algorithms.MAML(meta_agent.qnetwork_local, lr=5e-4)
    opt = torch.optim.Adam(maml.parameters(),lr=5e-5)

    #task_agent = deepcopy(meta_agent)
    task_agent1 = Agent(state_size=2, action_size=4, seed=0)

    task_agent2 = Agent(state_size=2, action_size=4, seed=0)

    task_agent3 = Agent(state_size=2, action_size=4, seed=0)

    # opt = torch.optim.Adam(meta_agent.qnetwork_local.parameters(), lr=LR_META)

    agents = [task_agent1,task_agent2,task_agent3]
    
    agent_epsilon = [1.0,1.0,1.0]
    eps_decay=0.995
    eps_min=0.001
    episodic_reward=[]
    scores_window = deque(maxlen=30)
    prev_state = task_agent1.qnetwork_local.state_dict()
    # opt_inner = torch.optim.Adam(task_agent1.qnetwork_local.parameters(), lr=LR_META)
    
    for i in range(TIMESTEPS):
        losses = [torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)]
        step_loss = torch.tensor(0.0)
        task_loss=0.0
        total_reward = 0.0
        epsiodes = 0
        
    
        for t in range(len(tasks)):
            env  = tasks[t]
            task_agent = agents[t]
            # Adaptation: Instantiate a copy of model
            # task_agent.qnetwork_local = deepcopy(meta_agent.qnetwork_local)
            # states = meta_agent.qnetwork_local.state_dict()
            # states = {k.replace('module.',''): v for k, v in states.items()}
            # task_agent.qnetwork_local.load_state_dict(states)
            task_agent.qnetwork_local = maml.clone()

            
            epsilon = agent_epsilon[t]

            steps = 0
            # while steps<STEPS_PER_TASK:
            for epsiode_num in range(EPISODES_PER_TASK):
            #while True:
                epsiodes+=1
                state = env.reset()

                while True:
                    #env.render()
                    action = task_agent.act(state,epsilon)
                    next_state, reward, terminal, _ = env.step(action)


                    # step_loss = step_loss+task_agent.step(state, action, reward, next_state, terminal).clone()
                    # task_agent.step(state, action, reward, next_state, terminal)
                    # losses[t] = losses[t]+task_agent.step(state, action, reward, next_state, terminal).clone()
                    step_loss = step_loss +task_agent.step(state, action, reward, next_state, terminal)

                    total_reward += reward
                    state = next_state
                    steps+=1

                    # Stopping condition for episode - agent encounters terminal state
                    if terminal or steps >= STEPS_PER_TASK:
                    # if terminal:
                        agent_epsilon[t] = max(eps_min, eps_decay*agent_epsilon[t])
                        # losses.append(deepcopy(task_loss))

                        break
                
                
                # if task_loss!=0.0:
                #     step_loss = step_loss+task_loss
                #     # task_agent.optimizer.zero_grad()
                #     # task_loss.backward(retain_graph=True)
                #     # task_agent.optimizer.step()
                #     # agents[t] = task_agent
            


            # step_loss+=task_agent.calculate_loss(int(STEPS_PER_TASK/2))
            #print("Step {} :: Task {} :: Loss {}".format(i,t,step_loss))

            # Take a gradient step on the loss and updates the cloned parameters in place
            # task_agent.current_network.adapt(step_loss)
            
            # Adaptation: Evaluate the effectiveness of adaptation
            # adapt_loss = task_agent.calculate_loss(int(BATCH_SIZE))

            # Accumulate the error over all tasks
            # step_loss += adapt_loss


        # Meta-learning step: compute gradient through the adaptation step, automatically.

        # step_loss = losses[0]+losses[1]+losses[2]
        step_loss = step_loss / (3)
        avg_reward = total_reward/(epsiodes)
        episodic_reward.append(avg_reward)
        scores_window.append(avg_reward)
        if (step_loss!=0.0 ):
            
            opt.zero_grad()
            step_loss.backward()
            opt.step()
            
            print('\r',i, "Average episodic reward",np.mean(scores_window),", Step loss:",step_loss.item(),'Epsilon: ',epsilon, end="")
            if i%100==0:
                print(i, "Average episodic reward",np.mean(scores_window),", Step loss:",step_loss.item(),'Epsilon: ',epsilon)

        if np.mean(np.array(scores_window))>=0.3 and i>200:
            torch.save(meta_agent.qnetwork_local.state_dict(),'./meta_3_3x3.pth')
            np.save('./avg_rewards_meta3.npy',np.array(episodic_reward))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(np.arange(len(episodic_reward)), episodic_reward)
            plt.ylabel('Score')
            plt.xlabel('Episode #')
            plt.savefig('meta3_exp_replay_training_result.png')
            return 0

        torch.save(meta_agent.qnetwork_local.state_dict(),'./meta3/meta_3_3x3iteration'+str(i)+'.pth')
        np.save('./meta3/avg_rewards_meta3.npy',np.array(episodic_reward))

        # if (i%50==0 and i>0):
        #     # print(prev_state['fc1.weight']==task_agent.qnetwork_local.state_dict()['fc1.weight'])
        #     prev_state=task_agent.qnetwork_local.state_dict()
    
    # plot the scores



if __name__ == '__main__':
    main()


