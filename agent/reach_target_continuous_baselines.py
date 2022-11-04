"""Note:
stablebaseline3 is required to run this examples
"""

import time
from datetime import datetime
from tqdm import tqdm

import gym
import airsim_gym
from stable_baselines3 import PPO,DQN,SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from common.callbacks import RewardCallback
from common.utils import arg_parser, save_model

args = arg_parser()


if __name__ == "__main__":

    eval_env = gym.make(
                    "airsim-gym-reach-target-continuous-v0",
                    target_x_movement_range=0,
                    target_y_movement_range=0,
                    target_z_movement_range=0,
                    target_yaw_offset=0,
                    target_pitch_offset=0,
                    target_roll_offset=0,
                    target_yaw_range=0,
                    target_pitch_range=0,
                    target_roll_range=48,
                    accident_reward= 10,
                    success_reward= 6000,
                    time_or_distance_limit_passed_reward=-2,
                    distance_coefficient = 5
                )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"running on '{device}'")

    env = DummyVecEnv(
        [
            lambda: Monitor(
                eval_env
            )
        ]
    )

    if args.mode == 'train':
        
        if args.algorithm == 'ppo':

            net = [128 for _ in range(9)]

            # policy_kwargs = dict(net_arch=[dict(pi=[256,256,256], vf=[256,256,256])])
            # policy_kwargs = dict(net_arch=net)
            policy_kwargs = dict(net_arch=[dict(pi=net, vf=net)])

            model = PPO(
                    "MultiInputPolicy",
                    env,
                    verbose=1,
                    device=device,
                    tensorboard_log="./tb_logs/",
                    policy_kwargs=policy_kwargs,
                    batch_size=512,

                )

            callbacks = []
            eval_callback = EvalCallback(
                env,
                callback_on_new_best=None,
                n_eval_episodes=5,
                best_model_save_path="./models/best_models/continuous/ppo/",
                log_path=".",
                eval_freq=10000,
            )
            callbacks.append(eval_callback)

            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/checkpoints/continuous',
                                                    name_prefix='ppo')

            callbacks.append(checkpoint_callback)
            callbacks.append(RewardCallback())

            kwargs = {}
            kwargs["callback"] = callbacks

            if args.load == '1':
                print('loading best model')
                model = PPO.load("./models/best_models/continuous/ppo/best_model",env)
            elif args.load:
                print(f"loading model from {args.load}")
                model = PPO.load(args.load,env)


            model.batch_size=512
            print(model.policy)
            model.clip_range = get_schedule_fn(100)

            model.learn(
                total_timesteps=args.timesteps,
                tb_log_name="ppo_airsim_drone_run_" + str(time.time()),
                **kwargs
            )

            # model.save("ppo_continuous_airsim_drone_policy")
            save_model(model, "models/train_finished/ppo_continuous_airsim_drone_policy")
        
        elif args.algorithm == 'sac':

            net = [256 for _ in range(8)]

            # policy_kwargs = dict(net_arch=[dict(pi=net, vf=net)])
            policy_kwargs = dict(net_arch=net)

            model = SAC(
                "MultiInputPolicy",
                env,
                learning_rate=0.0001,
                buffer_size=100000,
                verbose=1,
                device=device,
                policy_kwargs=policy_kwargs,
                tensorboard_log="./tb_logs/",
            )

            callbacks = []
            eval_callback = EvalCallback(
                env,
                callback_on_new_best=None,
                n_eval_episodes=5,
                best_model_save_path=f"./models/best_models/continuous/sac/{time.strftime('%Y-%m-%d_%H-%M-%S')}",
                log_path=".",
                eval_freq=10000,
            )
            callbacks.append(eval_callback)
            callbacks.append(RewardCallback())

            checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./models/checkpoints/continuous',
                                                    name_prefix='sac')

            callbacks.append(checkpoint_callback)

            kwargs = {}
            kwargs["callback"] = callbacks

            if args.load == '1':
                print('loading best model')
                model = SAC.load("./models/best_models/continuous/sac/best_model",env)
            elif args.load:
                print(f"loading model from {args.load}")
                model = SAC.load(args.load,env)

            if args.replay:
                print(f"loading replay buffer from {args.replay}")
                model.load_replay_buffer(args.replay)

            print(model.policy)

            model.gamma = 0.991
            model.batch_size = 1024

            print(model.gamma)
            print(model.batch_size)

            try:
                model.learn(
                    total_timesteps=args.timesteps,
                    tb_log_name="sac_continuous_airsim_drone_run_" + time.strftime('%Y-%m-%d_%H-%M-%S'),
                    **kwargs
                )
            except:
                t = time.strftime("%Y-%m-%d_%H-%M-%S")
                # print(f"saving data ...")
                # model.save(f"latest/sac_{t}")
                # model.save_replay_buffer(f"latest/sac_{t}_rep")

                save_model(model, f"models/run_stopped/sac_{t}")

            # model.save("sac_continuous_airsim_drone_policy")
            # model.save_replay_buffer("sac_continuous_airsim_drone_policy_replay_buffer")

            save_model(model,"models/train_finished/sac_continuous_airsim_drone_policy")

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
            if args.load == '1':
                print('loading best model')
                model = SAC.load("./best_models/continuous/sac/best_model",env)
            elif args.load:
                print(f"loading model from {args.load}")
                model = SAC.load(args.load,env)

            episodes = 35

            success = 0

            for e in tqdm(range(episodes)):
                obs = eval_env.reset()

                while True:
                    action , _state = model.predict(obs)
                    obs, reward, done, info = eval_env.step(action)

                    if done:
                        print(info['success'])
                        if info['success']:
                            success += 1
                        break

            print(f"number of success {success} | total {episodes} | success rate {(success/episodes)*100}")