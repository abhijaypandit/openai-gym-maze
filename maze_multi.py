import gym
from gym_maze import *

from stable_baselines3 import DQN, PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

train = False

if __name__ == '__main__':
    env_id = 'maze-sample-5x5-v0'
    num_cpu = 4  # Number of processes to use

    # Create the vectorized environment
    #env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv)

    # Instantiate the agent
    model = A2C('MlpPolicy',
                env=env,
                learning_rate=0.001,
                #exploration_fraction=0.05,
                verbose=1,
                tensorboard_log=env_id
            )

    if train:
        # Train the agent
        model.learn(total_timesteps=2e5)
        # Save the agent
        model.save(env_id)
        # Delete environment
        del env

    env = gym.make(env_id)

    obs = env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            env.reset()
