"""Note:
stablebaseline3 is required to run this examples
"""

import argparse
import time

import gym
import airsim_gym
from stable_baselines3 import PPO,DQN,HER,A2C
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

parser = argparse.ArgumentParser(description="baseline algorithms for traning and evaluating on airsim_gym environment.\
                                            Note that you should have stable basline3 installed \
                                            (use 'pip install stable-baselines3').")

parser.add_argument(
    "mode",
    choices=['train','eval'],
    help="select training or evaluating"
)

parser.add_argument(
    "algorithm",
    choices=['ppo','dqn','her','a2c'],
    help="which RL algorithm you want for train or evaluting"
)

parser.add_argument(
    '-t',
    '--timesteps',
    type=int,
    default=5e5,
    help="Number of timesteps of training. default=5e5."
)

parser.add_argument(
    '-l',
    '--load',
    help="load weights or train from scratch. {1} for best model or enter saved model path."
)

args = parser.parse_args()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"running on '{device}'")

    env = DummyVecEnv(
        [
            lambda: Monitor(
                gym.make(
                    "airsim-gym-reach-target-v0",
                    target_x_movement_range=0.1,
                    target_y_movement_range=0.1,
                    target_z_movement_range=0.1,
                )
            )
        ]
    )

    if args.mode == 'train':
        
        if args.algorithm == 'ppo':

            policy_kwargs = dict(net_arch=[dict(pi=[128, 256, 128], vf=[128, 256, 128])])

            model = PPO(
                    "MultiInputPolicy",
                    env,
                    verbose=1,
                    device=device,
                    tensorboard_log="./tb_logs/",
                    policy_kwargs=policy_kwargs
                )

            callbacks = []
            eval_callback = EvalCallback(
                env,
                callback_on_new_best=None,
                n_eval_episodes=5,
                best_model_save_path="./best_models/ppo/",
                log_path=".",
                eval_freq=10000,
            )
            callbacks.append(eval_callback)

            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                                    name_prefix='ppo')

            callbacks.append(checkpoint_callback)

            kwargs = {}
            kwargs["callback"] = callbacks

            if args.load == '1':
                print('loading best model')
                model.load("./best_models/ppo/best_model",env)
            elif args.load:
                print(f"loading model from {args.load}")
                model.load(args.load,env)


            model.learn(
                total_timesteps=args.timesteps,
                tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
                **kwargs
            )

            model.save("ppo_airsim_drone_policy")
        
        elif args.algorithm == 'dqn':

            policy_kwargs = dict(net_arch=[256,256],)

            model = DQN(
                "MultiInputPolicy",
                env,
                verbose=1,
                device=device,
                tensorboard_log="./tb_logs/",

                learning_rate=0.00025,
                batch_size=128,
                train_freq=2,
                target_update_interval=100,
                learning_starts=200,
                buffer_size=5000,
                max_grad_norm=10,
                exploration_fraction=0.1,
                exploration_final_eps=0.01
            )

            callbacks = []
            eval_callback = EvalCallback(
                env,
                callback_on_new_best=None,
                n_eval_episodes=5,
                best_model_save_path="./best_models/dqn/",
                log_path=".",
                eval_freq=10000,
            )
            callbacks.append(eval_callback)

            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/',
                                                    name_prefix='dqn')

            callbacks.append(checkpoint_callback)

            kwargs = {}
            kwargs["callback"] = callbacks

            if args.load == '1':
                print('loading best model')
                model.load("./best_models/dqn/best_model",env)
            elif args.load:
                print(f"loading model from {args.load}")
                model.load(args.load,env)

            model.learn(
                total_timesteps=args.timesteps,
                tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
                **kwargs
            )
