import cv2
import gym
import numpy as np
from gym import spaces
from PIL import Image

from gym.wrappers.monitoring.video_recorder import VideoRecorder
# https://github.com/chris-chris/mario-rl-tutorial/

class MyDownSampleWrapper(gym.ObservationWrapper):
    def __init__(self, env, image_size):
        super(MyDownSampleWrapper, self).__init__(env)
        self._image_size = image_size
        # set up a new observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._image_size[1], self._image_size[0], 1),
            dtype=np.uint8
        )

    def observation(self, frame):
        # convert the frame from RGB to gray scale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # resize the frame to the expected shape.
        #cv2.INTER_AREA
        #INTER_NEAREST
        frame = cv2.resize(frame, self._image_size)
        #frame = cv2.resize(frame, self._image_size)

        return frame[:, :, np.newaxis]


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

class MyRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self._x_position = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        _x_position = self.env.unwrapped._get_x_position()
        _reward = _x_position - self._x_position
        self._x_position = _x_position
        # resolve an issue where after death the x position resets. The x delta
        # is typically has at most magnitude of 3, 5 is a safe bound
        if _reward < -5 or _reward > 5:
            _reward = 0


        reward = _reward

        #print('reward', reward)

        return obs, reward, done, info

class CroppingWrapper(gym.ObservationWrapper):
    """
    """

    def __init__(self, env=None):
        super(CroppingWrapper, self).__init__(env)
        self.env = env

        self.new_size = (20,20)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.new_size[1], self.new_size[0], 1),
            dtype=np.uint8
        )

    def _observation(self, obs):
        #x_t = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        #x_t = np.reshape(x_t, (84, 84, 1))
        #x_t = np.nan_to_num(x_t)
        #return x_t.astype(np.uint8)
        #obs.resize((32,32))
        #size = (16,16)
        #obs.resize(size)

        obs = obs[10:30, 10:30, :] # 32x32 -> 20x20

        obs = np.array(obs)
        #obs = obs.shape((20,20,1))


        #obs2 = np.copy(obs)
        #obs2.resize(self.new_size)


        #img = Image.fromarray(obs)
        #cv2.imshow('image', obs2)
        #cv2.waitKey(1)
        #img.save('testaww_{}.png'.format(self.new_size))

        return obs

