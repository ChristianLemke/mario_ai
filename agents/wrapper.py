import cv2
import gym
import numpy as np
from gym import spaces

# https://github.com/chris-chris/mario-rl-tutorial/

class ProcessFrame84(gym.ObservationWrapper):
  def __init__(self, env=None):
    super(ProcessFrame84, self).__init__(env)
    self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

  def _observation(self, obs):
    return ProcessFrame84.process(obs)

  @staticmethod
  def process(img):
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    x_t = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
    x_t = np.reshape(x_t, (84, 84, 1))
    x_t = np.nan_to_num(x_t)
    return x_t.astype(np.uint8)