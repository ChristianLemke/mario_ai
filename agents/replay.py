import gym
import os
from baselines import deepq
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import pylab

from agents.wrapper import ProcessFrame84

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger

import datetime
import os

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_obs(obs):

    #img = Image.fromarray(obs, 'L')
    #img.show()


    #pylab.imshow(obs, cmap=pylab.gray())
    #pylab.show()

    #plt.imshow(obs, cmap='gray')
    #plt.show()

    #arr = np.random.randint(0, 256, 100*100)
    arr = obs.flatten()
    arr.resize((84, 84))
    arr.astype(np.uint8)
    im = Image.fromarray(arr, mode="L")
    #im.save("aa.png")
    im.show()



def main():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = ProcessFrame84(env)
    act = deepq.load(PROJ_DIR+"/../models/mario_model_2018-08-04T15:14:32.127429.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])

            plot_obs(obs)

            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':



    main()