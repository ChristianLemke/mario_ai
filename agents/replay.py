import gym
import os
from baselines import deepq
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv, DownsampleEnv, FrameStackEnv, PenalizeDeathEnv
from agents.wrapper import ProcessFrame84, FrameMemoryWrapper, VideoRecorderWrapper, EpisodicLifeEnv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pylab
from agents.wrapper import ProcessFrame84, FrameMemoryWrapper
from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger
import datetime
import time
import os

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

#set up environment and load trained models for replay....

def main():
    #env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v1')
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    timestart = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    # env = VideoRecorderWrapper(env, PROJ_DIR + "/../video", str(timestart), 50)

    env = VideoRecorderWrapper(env, PROJ_DIR + "/../video/final", str(timestart), 1)
    env = DownsampleEnv(env, (84, 84))
    env = PenalizeDeathEnv(env, penalty=-25)
    env = FrameStackEnv(env, 4)
    # good
    #act = deepq.load(PROJ_DIR+"/../models/mario_model_2018-08-12-13:00:58.pkl")

    # better
    act = deepq.load(PROJ_DIR + "/../models/mario_model_2018-08-12-19:21:50.pkl")
    
    episode = 0
    while True:
        obs, done = env.reset(), False
        stepnr = 0
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None])[0])

            if stepnr % 20 == 0:
                plot_obs(obs)

            episode_rew += rew
            stepnr += 1
        print("Episode reward", episode_rew, episode)
        episode = episode+1


if __name__ == '__main__':
    main()
