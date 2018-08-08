import cv2
import gym
import numpy as np
from gym import spaces

from gym.wrappers.monitoring.video_recorder import VideoRecorder
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


class VideoRecorderWrapper(gym.ObservationWrapper):
    """
    """

    def __init__(self, env=None, path=None, training_start= None, freq_episode=100):
        super(VideoRecorderWrapper, self).__init__(env)
        self.episode = 0
        self.env = env
        self.path= path
        self.training_start = training_start
        self.freq_episode = freq_episode
        self.rec = None
        self.rec_now = False

    def _observation(self, obs):
        if self.rec_now:
            self.rec.capture_frame()
        return obs

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.episode += 1

        if self.rec_now:
            print("Stop record episode {}".format(self.episode-1))
            self.rec.close()
            self.rec_now = False

        if self.episode % self.freq_episode == 0:
            print("Start record episode {}".format(self.episode))
            path = "{}/{}_{:0>5d}.mp4".format(self.path, self.training_start, self.episode)
            self.rec = VideoRecorder(self.env, path=path)
            self.rec_now = True



        return observation



class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped._get_life()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
            #print("was_real_done", self.env.unwrapped._get_life())
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
            #print("nicht was_real_done", self.env.unwrapped._get_life())
        self.lives = self.env.unwrapped._get_life()
        return obs