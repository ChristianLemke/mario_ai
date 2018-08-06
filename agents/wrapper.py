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



class FrameMemoryWrapper(gym.ObservationWrapper):
    """
    Use 4 Frames in observation space.
    """

    frame_queue = None
    def __init__(self, env=None):
        super(FrameMemoryWrapper, self).__init__(env)
        self.frame_nr = 4
        self.frame_queue = []
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, self.frame_nr))

    def _observation(self, obs):
        return FrameMemoryWrapper.process(obs, self.frame_queue)

    def _reset(self):
        super().reset()
        print('clear')
        self.frame_queue.clear()

    @staticmethod
    def process(img, frame_queue):
        # first step
        if len(frame_queue) == 0:
            frame_queue.append(img)
            frame_queue.append(img)
            frame_queue.append(img)

        frame_queue.append(img)


        obs = np.concatenate(frame_queue[-4:], axis=2)


        #x_t = np.reshape(x_t, (84, 84, 1))


        return obs.astype(np.uint8)
