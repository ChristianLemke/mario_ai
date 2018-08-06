from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from agents.wrapper import ProcessFrame84, FrameMemoryWrapper

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.cmd_util import atari_arg_parser

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    #env = make_atari(env_id)

    env = gym_super_mario_bros.make('SuperMarioBros-v1')
    # env = gym_super_mario_bros.make('SuperMarioBrosNoFrameskip-v3')

    env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
    env = ProcessFrame84(env)

    env = FrameMemoryWrapper(env)




    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    #env = wrap_deepmind(env)
    env.seed(workerseed)


    def render_callback(lcl, _glb):
        # print(lcl['episode_rewards'])
        total_steps = lcl['env'].total_steps
        #if total_steps % 1000 == 0:
        #    print("Saving model to mario_model.pkl")
        #    act.save("../models/mario_model_{}.pkl".format(modelname))


        env.render()
        # pass


    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(num_timesteps * 1.1),
        timesteps_per_actorbatch=2048,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4,
        optim_stepsize=1e-3, # 3e-4
        optim_batchsize=64, #256
        gamma=0.99, lam=0.95,
        schedule='linear',
        callback = render_callback
    )
    env.close()

def main():
    args = atari_arg_parser().parse_args()



    train(args.env, num_timesteps=int(10e4), seed=args.seed)

if __name__ == '__main__':
    main()