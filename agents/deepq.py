from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv, DownsampleEnv, FrameStackEnv, PenalizeDeathEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from agents.wrapper import FrameMemoryWrapper, VideoRecorderWrapper, EpisodicLifeEnv
from baselines import deepq
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import WarpFrame, FrameStack
from baselines import bench
from baselines import logger
import argparse
import datetime
import time
import os
import sys

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dueling', type=int, default=0)
    #parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-path', type=str, default='/.')

    args = parser.parse_args()
    # TODO change logging dir for tensorboard
    #logger.configure(dir=None, format_strs='stdout,log,csv,json,tensorboard')
    #logger.configure(dir=None, format_strs=['stdout', 'log', 'csv', 'json', 'tensorboard'])
    timestart = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M:%S')
    logger.configure(dir=PROJ_DIR+"/../tensorboard/"+str(timestart), format_strs=['stdout','log','csv','json','tensorboard'])
    logger.set_level(logger.INFO)
    set_global_seeds(args.seed)

    env = gym_super_mario_bros.make('SuperMarioBros-v1')
#wrap environment 
    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    
#record videos of an episode
    env = VideoRecorderWrapper(env, PROJ_DIR+"/../video", str(timestart), 50)
#the agent has only one trial 
    env = EpisodicLifeEnv(env)

    # nes_py
    #preprocess the input frame
    env = DownsampleEnv(env, (84, 84))
    #set death penalty
    env = PenalizeDeathEnv(env, penalty=-25)
    #Stack 4 Framse as input
    env = FrameStackEnv(env, 4)
    
    #print tensorboard log information
    print("logger.get_dir():", logger.get_dir())
    print("PROJ_DIR:", PROJ_DIR)

    act = None
    #enable output in the terminal 
    env = bench.Monitor(env, logger.get_dir())

    modelname = datetime.datetime.now().isoformat()
    
    #define callback function for the training process
    def render_callback(lcl, _glb):
        # print(lcl['episode_rewards'])
        total_steps = lcl['env'].total_steps
        #if total_steps % 2000 == 0:

        env.render()
        # pass

#different models with different parameters. out commented
#CNN built deepq.models.with cnn_to_mlp(params)
#trained with deepq.learn(params)

    # model 01
    '''
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],  # (num_outputs, kernel_size, stride)
        hiddens=[256],
        dueling=bool(args.dueling),
    )
    '''

    # model 02
    # like the deep mind paper
    # First layer input 84x84x4
    # The first hidden layer convolves 16 8x8 filters, 4 stride
    # The second hidden layer convolves 32 4x4 filters, 2 stride
    # The flast layer 256 neurons
    '''
    model = deepq.models.cnn_to_mlp(
        convs=[(16, 8, 4), (32, 4, 2)],  # (num_outputs, kernel_size, stride)
        hiddens=[256],
        dueling=bool(0),
    )
    '''


    # model 03
    # nature human paper
    # The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the input image and applies a rectifier nonlinearity.
    # The second hidden layer convolves 64 filters of 4 x 4 with stride 2, again followed by a rectifier nonlinearity.
    # This is followed by a third convolutional layer that convolves 64 filters of 3 x 3 with stride 1 followed by a rectifier.
    # The final hidden layer is fully-connected and consists of 512 rectifier units.
    '''
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],  # (num_outputs, kernel_size, stride)
        hiddens=[512],
        dueling=bool(0),
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=0.00025,  # 1e-4
        max_timesteps=int(100000),
        buffer_size=50000,  # 5000, #10000
        exploration_fraction=0.9,  # 0.1,
        exploration_final_eps=0.1,  # 0.01
        train_freq=4,  # 4
        learning_starts=25000,  # 10000
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(0),
        prioritized_replay_alpha=0.6,
        checkpoint_freq=args.checkpoint_freq,
        #        checkpoint_path=args.checkpoint_path,
        callback=render_callback,
        print_freq=1
    )
    '''

#2:20 model 03 max_timesteps=int(1000000), buffer_size=50000,
#2:21 model 03 max_timesteps=int(500000), buffer_size=25000,
# 2018-08-07-11:49:49 model 04
# 2018-08-07-14:56:07_00800 model 4, 300k timesteps
# 2018-08-07-22:10:37 v3, model 4, 200k timesteps
# 2018-08-07-22:10:42 v1, model 4, 200k timesteps
#2018-08-12-08:38:40 model 4, 50k timesteps, lr 0.5, ef 0.9, gamma 0.95 v1
#2018-08-12-09:19:56 model 4, 100k, lr 0.0025, alpha 0.8, gamma 0.9, 8 frames v1
#2018-08-12-12:56:52 model 4, 100k, lr 0.0025, alpha 0.8, gamma 0.9, 4 frames v1
#2018-08-12-12:58:06 model 4, 100k, lr 0.0001, alpha 0.8, gamma 0.9, 4 frames v1
#2018-08-12-12:59:49 model 4, 100k, lr 0.0001, alpha 0.5, gamma 0.9, 4 frames v1
#2018-08-12-13:00:58 model 4, 100k, lr 0.0001, alpha 0.2, gamma 0.9, 4 frames v1
# last day
# best
# 2018-08-12-16:45:48 model 4, 100k, lr 0.0001, alpha 0.2, gamma 0.9, 4 frames v1 hidden 256

# 2018-08-12-17:48:48 gamma=0.2 fraction=0.3, lr= 0.0001, hidden 256

# 2018-08-12-18:12:12 gamma=0.5, fraction=0.3, lr= 0.0001, hidden 256

# 2018-08-12-19:21:50 512

#2018-08-12-10:25:50 model 4, 100k, lr 0.0005, alpha 0.6, gamma 0.99, 8 frames v1
#2018-08-12-11:31:59 model 4, 100k, lr 0.0005, alpha 0.8, gamma 0.99, 6 frames v1

# model 04
# nature human paper + Improvements
# Dueling Double DQN, Prioritized Experience Replay, and fixed Q-targets
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],  # (num_outputs, kernel_size, stride)
        hiddens=[512],# 512
        dueling=bool(1),
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr= 0.0001,  # 0.00025 1e-4
        max_timesteps=int(100000), # 100k -> 3h
        buffer_size=50000,  # 5000, #10000
        exploration_fraction=0.3,  # 0.1,
        exploration_final_eps=0.1,  # 0.01
        train_freq=4,  # 4
        learning_starts=25000,  # 10000
        target_network_update_freq=1000,
        gamma=0.5, #0.99,
        prioritized_replay=bool(1),
        prioritized_replay_alpha=0.2,
        checkpoint_freq=args.checkpoint_freq,
        #        checkpoint_path=args.checkpoint_path,
        callback=render_callback,
        print_freq=1
    )


    # 2018-08-08-18:38:35, model4 100k, e_f 0.9, e_f_eps 0.1, trainf 4, gamma 0.95, replay 1, lr 0.00025
    # 2018-08-08-19:29:19, model4 200k, e_f 0.9, e_f_eps 0.1, trainf 4, gamma 0.95, replay 1, lr 0.00025
    # 2018-08-08-19:29:54, model4 300k, e_f 0.9, e_f_eps 0.1, trainf 4, gamma 0.95, replay 1, lr 0.00025
    # 2018-08-08-20:25:59, model 4 100k, lr = 0.001
    # 2018-08-08-20:35:42  model 4 100k, lr = 0.05
    # 2018-08-08-20:35:42  model 4 100k, lr = 0.01 mode 3
    # 2018-08-08-22:37:00 model 4 50k, lr = 0.2 mode 1


    # model 05
    # nature human paper + Improvements
    # Dueling Double DQN, Prioritized Experience Replay, and fixed Q-targets
    '''
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4)],  # (num_outputs, kernel_size, stride)
        hiddens=[128, 64],
        dueling=bool(1),
    )

    act = deepq.learn(
        env,
        q_func=model,
        lr=0.5,  # 0.00025 1e-4
        max_timesteps=int(50000),  # 100k -> 3h
        buffer_size=50000,  # 5000, #10000
        exploration_fraction=0.9,  # 0.1,
        exploration_final_eps=0.1,  # 0.01
        train_freq=4,  # 4
        learning_starts=25000,  # 10000
        target_network_update_freq=1000,
        gamma=0.99,  # 0.99,
        prioritized_replay=bool(1),
        prioritized_replay_alpha=0.6,
        checkpoint_freq=args.checkpoint_freq,
        #        checkpoint_path=args.checkpoint_path,
        callback=render_callback,
        print_freq=1
    )
'''

    print("Saving model to mario_model.pkl " + timestart)
    act.save("../models/mario_model_{}.pkl".format(timestart))

    env.close()


if __name__ == '__main__':
    main()
