from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from agents.wrapper import ProcessFrame84

from baselines import deepq
from baselines.common import set_global_seeds
from baselines import bench
import argparse
from baselines import logger

import datetime
import os
import sys

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(10e4))
    #parser.add_argument('--checkpoint-freq', type=int, default=10000)
    parser.add_argument('--checkpoint-freq', type=int, default=1000)
    parser.add_argument('--checkpoint-path', type=str, default='/.')

    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)

    #env = make_atari(args.env)
    env = gym_super_mario_bros.make('SuperMarioBros-v1')
    #env = gym_super_mario_bros.make('SuperMarioBrosNoFrameskip-v3')


    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    env = ProcessFrame84(env)

    print("logger.get_dir():", logger.get_dir())
    print("PROJ_DIR:", PROJ_DIR)

    act = None

    env = bench.Monitor(env, logger.get_dir())
    #env = deepq.wrap_atari_dqn(env)
    model = deepq.models.cnn_to_mlp(
        #convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], # (num_outputs, kernel_size, stride)
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],  # (num_outputs, kernel_size, stride)
        hiddens=[256],
        dueling=bool(args.dueling),
    )

    modelname = datetime.datetime.now().isoformat()

    def render_callback(lcl, _glb):
        # print(lcl['episode_rewards'])
        total_steps = lcl['env'].total_steps
        #if total_steps % 1000 == 0:
        #    print("Saving model to mario_model.pkl")
        #    act.save("../models/mario_model_{}.pkl".format(modelname))


        env.render()
        # pass


    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-4,
        max_timesteps=args.num_timesteps,
        buffer_size=10000,
        exploration_fraction=0.5,#0.1,
        exploration_final_eps=0.01,
        train_freq=4,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=bool(args.prioritized),
        prioritized_replay_alpha=args.prioritized_replay_alpha,
        checkpoint_freq=args.checkpoint_freq,
#        checkpoint_path=args.checkpoint_path,
        callback=render_callback,
        print_freq=1
    )

    print("Saving model to mario_model.pkl")
    act.save("../models/mario_model_{}.pkl".format(datetime.datetime.now().isoformat()))

    env.close()



def deepq_callback(locals, globals):
  #pprint.pprint(locals)
  global max_mean_reward, last_filename
  if('done' in locals and locals['done'] == True):
    if('mean_100ep_reward' in locals
      and locals['num_episodes'] >= 10
      and locals['mean_100ep_reward'] > max_mean_reward
      ):
      print("mean_100ep_reward : %s max_mean_reward : %s" %
            (locals['mean_100ep_reward'], max_mean_reward))

      if(not os.path.exists(os.path.join(PROJ_DIR,'models/deepq/'))):
        try:
          os.mkdir(os.path.join(PROJ_DIR,'models/'))
        except Exception as e:
          print(str(e))
        try:
          os.mkdir(os.path.join(PROJ_DIR,'models/deepq/'))
        except Exception as e:
          print(str(e))

      if(last_filename != ""):
        os.remove(last_filename)
        print("delete last model file : %s" % last_filename)

      max_mean_reward = locals['mean_100ep_reward']
      act = deepq.ActWrapper(locals['act'], locals['act_params'])

      filename = os.path.join(PROJ_DIR,'models/deepq/mario_reward_%s.pkl' % locals['mean_100ep_reward'])
      act.save(filename)
      print("save best mean_100ep_reward model to %s" % filename)
      last_filename = filename


if __name__ == '__main__':
    main()