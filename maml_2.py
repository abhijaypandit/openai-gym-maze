from sqlite3 import adapt
import torch as th
from torch import nn, optim, distributions as dist
import gym
from gym_maze import *
import learn2learn as l2l
from copy import deepcopy
from agent import *
import matplotlib.pyplot as plt
from collections import deque

th.autograd.set_detect_anomaly(True)

DIM = 5
TIMESTEPS = 4000
TASKS_PER_STEP = 1

def main():
    env1 = gym.make('maze-random-3x3-v0')
    env2 = gym.make('maze-random-3x3-v0')
    env3 = gym.make('maze-random-3x3-v0')
    # tasks = [env1, env2, env3
    tasks = [env1]

    num_states=2
    num_actions=4
    
    ALPHA = 5e-4
    GAMMA = 0.99
    EPSILON = 1.0
    STEPS_PER_TASK = 100
    EPISODES_PER_TASK=1
    LR_META=5e-4
    # BATCH_SIZE = 32
    
    meta_agent =Agent(state_size=2, action_size=4, seed=0)

    #task_agent = deepcopy(meta_agent)
    task_agent1 = Agent(state_size=2, action_size=4, seed=0)

    task_agent2 = Agent(state_size=2, action_size=4, seed=0)

    task_agent3 = Agent(state_size=2, action_size=4, seed=0)
    maml = l2l.algorithms.MAML(meta_agent.qnetwork_local, lr=5e-4)
    opt = optim.Adam(meta_agent.qnetwork_local.parameters(), lr=LR_META)

    agents = [task_agent1,task_agent2,task_agent3]
    agent_epsilon = [1.0,1.0,1.0]
    eps_decay=0.9995
    eps_min=0.001
    episodic_reward=[]
    scores_window = deque(maxlen=50)
    for i in range(TIMESTEPS):
        step_loss = 0.0
        task_loss=0.0
        total_reward = 0.0
        epsiodes = 0
        
    
        for t in range(len(tasks)):
            task_loss = 0.0
            env  = tasks[t]
            task_agent = agents[t]
            # Adaptation: Instantiate a copy of model
            task_agent.qnetwork_local = maml.clone()
            opt = optim.Adam(meta_agent.qnetwork_local.parameters(), lr=LR_META)
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
                    task_loss = task_loss+task_agent.step(state, action, reward, next_state, terminal)



                    total_reward += reward
                    state = next_state
                    steps+=1

                    # Stopping condition for episode - agent encounters terminal state
                    if terminal or steps >= STEPS_PER_TASK:
                    # if terminal:
                        agent_epsilon[t] = max(eps_min, eps_decay*agent_epsilon[t])
                        break
                
                step_loss = step_loss+task_loss
                if task_loss>0.0:
                    task_agent.qnetwork_local.adapt(task_loss)

            # step_loss+=task_agent.calculate_loss(int(STEPS_PER_TASK/2))
            #print("Step {} :: Task {} :: Loss {}".format(i,t,step_loss))

            # Take a gradient step on the loss and updates the cloned parameters in place
            # task_agent.current_network.adapt(step_loss)
            
            # Adaptation: Evaluate the effectiveness of adaptation
            # adapt_loss = task_agent.calculate_loss(int(BATCH_SIZE))

            # Accumulate the error over all tasks
            # step_loss += adapt_loss


        # Meta-learning step: compute gradient through the adaptation step, automatically.
        step_loss = step_loss / (TASKS_PER_STEP)
        avg_reward = total_reward/(epsiodes)
        episodic_reward.append(avg_reward)
        scores_window.append(avg_reward)
        if (step_loss>0.0 ):
            
            opt.zero_grad()
            #step_loss.backward(retain_graph=True)
            step_loss.backward()
            opt.step()
            
            print(i, "Average episodic reward",avg_reward,", Step loss:",step_loss.item(),'Epsilon: ',epsilon)

        if np.mean(np.array(scores_window))>=0.93 and len(scores_window)>50:
            th.save(meta_agent.qnetwork_local.state_dict(),'./meta_2_3x3.pth')
            np.save('./avg_rewards_meta2.npy',np.array(episodic_reward))
            break
    
    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(episodic_reward)), episodic_reward)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('meta2_exp_replay_training_result.png')
    


if __name__ == '__main__':
    main()


