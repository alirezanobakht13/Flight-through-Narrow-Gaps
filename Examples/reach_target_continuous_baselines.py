"""Note:
stablebaseline3 is required to run this examples
"""

import argparse
from operator import mod
import time
from tqdm import tqdm

import gym
import airsim_gym
from stable_baselines3 import PPO,DQN,SAC
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
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
    choices=['ppo','sac'],
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
                    "airsim-gym-reach-target-continuous-v0",
                    target_x_movement_range=1,
                    target_y_movement_range=1,
                    target_z_movement_range=1,
                    accident_reward=-1,
                    success_reward= 50,
                    time_or_distance_limit_passed_reward=-2,
                    distance_coefficient = 5
                )
            )
        ]
    )

    if args.mode == 'train':
        
        if args.algorithm == 'ppo':

            policy_kwargs = dict(net_arch=[dict(pi=[256,256,256], vf=[256,256,256])])

            model = PPO(
                    "MultiInputPolicy",
                    env,
                    verbose=1,
                    device=device,
                    tensorboard_log="./tb_logs/",
                    policy_kwargs=policy_kwargs,
                    batch_size=128
                )

            callbacks = []
            eval_callback = EvalCallback(
                env,
                callback_on_new_best=None,
                n_eval_episodes=5,
                best_model_save_path="./best_models/continuous/ppo/",
                log_path=".",
                eval_freq=10000,
            )
            callbacks.append(eval_callback)

            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/continuous',
                                                    name_prefix='ppo')

            callbacks.append(checkpoint_callback)

            kwargs = {}
            kwargs["callback"] = callbacks

            if args.load == '1':
                print('loading best model')
                model = PPO.load("./best_models/continuous/ppo/best_model",env)
            elif args.load:
                print(f"loading model from {args.load}")
                model = PPO.load(args.load,env)


            model.learn(
                total_timesteps=args.timesteps,
                tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
                **kwargs
            )

            model.save("ppo_continuous_airsim_drone_policy")
        
        elif args.algorithm == 'sac':

            policy_kwargs = dict(net_arch=[dict(pi=[256,256,256], vf=[256,256,256])])

            model = SAC(
                "MultiInputPolicy",
                env,
                learning_rate=0.0001,
                buffer_size=100000,
                verbose=1,
                device=device,
                tensorboard_log="./tb_logs/",
            )

            callbacks = []
            eval_callback = EvalCallback(
                env,
                callback_on_new_best=None,
                n_eval_episodes=5,
                best_model_save_path="./best_models/continuous/sac/",
                log_path=".",
                eval_freq=10000,
            )
            callbacks.append(eval_callback)

            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints/continuous',
                                                    name_prefix='sac')

            callbacks.append(checkpoint_callback)

            kwargs = {}
            kwargs["callback"] = callbacks

            if args.load == '1':
                print('loading best model')
                model = SAC.load("./best_models/continuous/sac/best_model",env)
            elif args.load:
                print(f"loading model from {args.load}")
                model = SAC.load(args.load,env)


            model.learn(
                total_timesteps=args.timesteps,
                tb_log_name="sac_continuous_airsim_drone_run_" + str(time.time()),
                **kwargs
            )

            model.save("sac_continuous_airsim_drone_policy")

    if args.mode == 'eval':
        
        if args.algorithm == 'ppo':
            model = PPO.load('checkpoints/ppo_420000_steps',env)

            # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)

            # print(f"mean reward: {mean_reward} | std reward: {std_reward}")

            episodes = 50

            success = 0

            for e in tqdm(range(episodes)):
                obs = env.reset()

                while True:
                    action , _state = model.predict(obs)
                    obs, reward, done, info = env.step(action)

                    if done:
                        if reward == 50:
                            success += 1
                        break

            print(f"number of success {success} | total {episodes} | success rate {(success/episodes)*100}")

        elif args.algorithm == 'sac':
            model = SAC.load('checkpoints/continuous/sac_40000_steps')

            episodes = 10

            success = 0

            for e in tqdm(range(episodes)):
                obs = env.reset()

                while True:
                    action , _state = model.predict(obs)
                    obs, reward, done, info = env.step(action)

                    if done:
                        if reward == 50:
                            success += 1
                        break

            print(f"number of success {success} | total {episodes} | success rate {(success/episodes)*100}")