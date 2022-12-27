# --------------------------- Python related stuff --------------------------- #
from typing import Union, Optional, Dict, Text, Tuple, Callable
import json
import argparse
import os
import logging

# ------------------- Stable-baseline modules and functions ------------------ #
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

import torch

# ------------------------ Gym and custom environment ------------------------ #
import gym
import airsim_gym

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)


def arg_parser():
    """
    CLI arguments are defined here.
    """
    parser = argparse.ArgumentParser(description="baseline algorithms for training and evaluating on airsim_gym environment.\
                                            Note that you should have stable baseline3 installed \
                                            (use 'pip install stable-baselines3').")

    parser.add_argument(
        "mode",
        choices=['train', 'eval'],
        help="select training or evaluating"
    )

    parser.add_argument(
        '-a',
        '--algorithm',
        choices=['ppo', 'sac'],
        help="which RL algorithm you want for training or evaluating."
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

    parser.add_argument(
        '-r',
        '--replay',
        help="load replay buffer or train from scratch. {1} for best model or enter saved model path."
    )

    parser.add_argument(
        '-c',
        '--config',
        help="load config file from the given path, or load default config if nothing is given.\n\
            note that if the variable is in both arguments and config file, argument one is selected."
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        help="batch size of RL algorithms"
    )

    parser.add_argument(
        '-g',
        '--gamma',
        help="discount factor"
    )

    parser.add_argument(
        '--learning_rate',
        help="learning rate"
    )

    parser.add_argument(
        '--distance_coefficient',
        help="distance coefficient"
    )

    parser.add_argument(
        '--accident_reward',
        help="accident reward (it should be positive. it will be negated in program)."
    )

    parser.add_argument(
        '--success_reward',
        help="success reward."
    )

    args = parser.parse_args()
    return args


def source_code_correction(code: str, is_w3: bool) -> str:
    index = code.find('def')
    new_code = ''
    for l in iter(code.splitlines()):
        new_code += l[index:] + '\n'

    par_index = new_code.find('(')

    if is_w3:
        new_code = new_code[:4] + 'w3_calc' + new_code[par_index:]
    else:
        new_code = new_code[:4] + 'w4_calc' + new_code[par_index:]

    return new_code


def save_model(
    model: Union[PPO, DQN, SAC],
    path: Text,
    config: Optional[Dict] = None
) -> None:
    """
    save the model to given path.
    config could contain information about:
    - environment:
        - reward weights
        - target movement
        - ...
    - model:
        - network architecture
        - gamma
        - learning rate
        - ...

    save pattern is:
    - model: <path>/model.zip
    - replay_buffer: <path>/replay_buffer.pkl (if model is DQN or SAC)
    - config: <path>/config_model_env.py (if config is given)
    """

    if not os.path.exists(path):
        os.makedirs(path)

    logging.info(f"saving model to {path}/model")
    model.save(f"{path}/model")

    if isinstance(model, SAC) or isinstance(model, DQN):
        logging.info(f"save model replay buffer to {path}/replay_buffer")
        model.save_replay_buffer(f"{path}/replay_buffer")

    if config:
        w3_calc_fn, w4_calc_fn = '', ''

        if 'w3_calc_fn' in config['environment']:
            if get(config, 'environment', 'w3_calc_fn'):
                w3_calc_fn = config['environment']["w3_calc_fn"]
                w3_calc_fn = source_code_correction(w3_calc_fn, True)
            config['environment'].pop('w3_calc_fn')

        if 'w4_calc_fn' in config['environment']:
            if get(config, 'environment', 'w4_calc_fn'):
                w4_calc_fn = config['environment']["w4_calc_fn"]
                w4_calc_fn = source_code_correction(w4_calc_fn, False)
            config['environment'].pop('w4_calc_fn')

        logging.info(f"save model config to {path}/config_model_env")
        with open(f"{path}/config_model_env.py", "w") as f:
            f.write("config = " + json.dumps(config, indent=4)
                    .replace('false', 'False').replace('true', 'True').replace('null', 'None'))
            f.write("\n\n")
            f.write(w3_calc_fn)
            f.write("\n\n")
            f.write(w4_calc_fn)
            f.write("\n")


def load_config_file(path: str) -> Tuple[Dict, Optional[Callable], Optional[Callable]]:
    config = None

    if os.path.exists(f"{path}"):
        print(path)
        from importlib.machinery import SourceFileLoader
        config = SourceFileLoader(os.path.splitext(
            os.path.basename(path))[0], path).load_module()

    if not config:
        return None, None, None

    return config.config, (config.w3_calc if 'w3_calc' in dir(config) else None), \
        (config.w4_calc if 'w4_calc' in dir(config) else None)


def get(name: Union[dict, None], *args: str):
    """
    nested get. None is returned if last or each of middle elements doesn't exist.
    """
    if name is None:
        return None
    for i in args:
        name = name.get(i, None)
        if name is None:
            return None
    return name


def namespace_get(namespace, attr):
    return namespace.__getattribute__(attr) if attr in dir(namespace) else None


def load_from_path(
        path: Text,
        algorithm: Optional[Text] = None
) -> Tuple[Union[PPO, DQN, SAC],
           Optional[Dict],
           Optional[Callable],
           Optional[Callable]]:
    """
    load model (and config if exists) from given path.
    Note that algorithm of model should be defined.
    pattern is like save_model function:
    - model: <path>/model.zip
    - replay buffer: <path>/replay_buffer.pkl (if exists and algorithm is DQN or SAC)
    - config: <path>/config_model_env.py (if exits)
    """
    if not os.path.exists(f"{path}/model.zip"):
        raise Exception("model doesn't exist.")

    config, w3_calc, w4_calc = load_config_file(f"{path}/config_model_env.py")

    if config and config.get('model', None):
        algorithm = config['model'].get('algorithm', None) or algorithm

    if algorithm is None:
        raise Exception("RL algorithm is not defined.\
            it should be given by CLI argument or exists in config file")

    if algorithm == 'dqn':
        model = DQN.load(f"{path}/model")
        logging.info(f"DQN model is loaded from {path}/model.zip")
        if os.path.exists(f"{path}/replay_buffer.pkl"):
            model.load_replay_buffer(f"{path}/replay_buffer")
            logging.info(
                f"DQN replay buffer is loaded from {path}/replay_buffer.pkl")

    elif algorithm == 'ppo':
        model = PPO.load(f"{path}/model")
        logging.info(f"PPO model is loaded from {path}/model.zip")

    elif algorithm == "sac":
        model = SAC.load(f"{path}/model")
        logging.info(f"SAC model is loaded from {path}/model.zip")
        if os.path.exists(f"{path}/replay_buffer.pkl"):
            model.load_replay_buffer(f"{path}/replay_buffer")
            logging.info(
                f"SAC replay buffer is loaded from {path}/replay_buffer.pkl")

    return model, config, w3_calc, w4_calc


def setup():
    """
    setting up environment and model.
    config of environment and model will be selected in the following order:
    1 - argument given by CLI
    2 - config file given in CLI
    3 - config file loaded when loading model
    4 - default config file
    5 - None. default value of model and environment itself will be chosen if available, error will be thrown otherwise.
    """
    environment_parameters = ['id', 'ip_address', 'port', 'movement_size', 'max_distance', 'target_init_x',
                              'target_init_y', 'target_init_z', 'target_x_movement_range',
                              'target_y_movement_range', 'target_z_movement_range', 'target_yaw_offset',
                              'target_pitch_offset', 'target_roll_offset', 'target_yaw_range',
                              'target_pitch_range', 'target_roll_range', 'max_timestep', 'accident_reward',
                              'success_reward', 'time_or_distance_limit_passed_reward', 'distance_coefficient']

    model_parameters = ['algorithm', 'gamma', 'learning_rate',
                        'batch_size', 'policy_kwargs', 'tensorboard_log']

    args = arg_parser()

    config_c, w3_calc_c, w4_calc_c = None, None, None

    if args.config:
        config_c, w3_calc_c, w4_calc_c = load_config_file(args.config)

    config_d, w3_calc_d, w4_calc_d = load_config_file("../default_config.py")

    config_l, w3_calc_l, w4_calc_l = None, None, None
    model = None

    if args.load:
        model, config_l, w3_calc_l, w4_calc_l = load_from_path(args.load)

    main_config = {
        "environment": {},
        "model": {}
    }

    for env_var in environment_parameters:
        temp = None
        if namespace_get(args, env_var) is not None:
            temp = namespace_get(args, env_var)
        elif get(config_c, 'environment', env_var) is not None:
            temp = get(config_c, 'environment', env_var)
        elif get(config_l, 'environment', env_var) is not None:
            temp = get(config_l, 'environment', env_var)
        elif get(config_d, 'environment', env_var) is not None:
            temp = get(config_d, 'environment', env_var)

        if temp is not None:
            main_config['environment'][env_var] = temp

    for model_var in model_parameters:
        temp = None
        if namespace_get(args, model_var) is not None:
            temp = namespace_get(args, model_var)
        elif get(config_c, 'model', model_var) is not None:
            temp = get(config_c, 'model', model_var)
        elif get(config_l, 'model', model_var) is not None:
            temp = get(config_l, 'model', model_var)
        elif get(config_d, 'model', model_var) is not None:
            temp = get(config_d, 'model', model_var)

        if temp is not None:
            main_config['model'][model_var] = temp

    main_config['environment']['w3_calc_fn'] = w3_calc_c or w3_calc_l or w3_calc_d or None
    main_config['environment']['w4_calc_fn'] = w4_calc_c or w4_calc_l or w4_calc_d or None

    logging.info(f"config = {json.dumps(main_config, indent=4)}")

    env = gym.make(**main_config['environment'])
    envs = DummyVecEnv(
        [
            lambda: Monitor(
                env
            )
        ]
    )

    if not args.load:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        if main_config['model']['algorithm'] == 'sac':
            main_config['model'].pop('algorithm')

            model = SAC(
                "MultiInputPolicy",
                envs,
                verbose=namespace_get(args, 'verbose') or 1,
                device=device,
                **main_config['model']
            )
            main_config['model']['algorithm'] = 'sac'

        elif main_config['model']['algorithm'] == 'ppo':
            main_config['model'].pop('algorithm')
            model = PPO(
                envs,
                verbose=namespace_get(args, 'verbose') or 1,
                device=device,
                **main_config['model']
            )
            main_config['model']['algorithm'] == 'ppo'

        else:
            raise Exception("Neither model is loaded nor model is given.")

    else:
        for model_var in main_config['model']:
            model[model_var] = main_config['model'][model_var]

    return model, env, envs, main_config, args
