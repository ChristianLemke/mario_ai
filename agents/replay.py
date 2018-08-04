import gym
import os
from baselines import deepq
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agents.wrapper import ProcessFrame84

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger

import datetime
import os

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = ProcessFrame84(env)
    act = deepq.load(PROJ_DIR+"/../models/mario_model_2018-08-03T23:05:17.361042.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()