from typing import Union, Optional, Dict, Text, Tuple, Callable
import json
import argparse
import os

from stable_baselines3 import PPO, DQN, SAC


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
        '--lr',
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

    print(f"saving model to {path}/model")
    model.save(f"{path}/model")

    if isinstance(model, SAC) or isinstance(model, DQN):
        print(f"save model replay buffer to {path}/replay_buffer")
        model.save_replay_buffer(f"{path}/replay_buffer")

    if config:

        w3_calc_fn = config.pop("w3_calc_fn")
        w3_calc_fn = source_code_correction(w3_calc_fn, True)
        w4_calc_fn = config.pop("w4_calc_fn")
        w4_calc_fn = source_code_correction(w4_calc_fn, False)

        algo = None
        if isinstance(model, SAC):
            algo = 'sac'
        elif isinstance(model, PPO):
            algo = 'ppo'
        elif isinstance(model, DQN):
            algo = 'dqn'

        setting = {
            "environment": config,
            "model": {
                "algorithm": algo,
                "policy_kwargs": model.policy_kwargs
            }
        }

        print(f"save model config to {path}/config_model_env")
        with open(f"{path}/config_model_env.py", "w") as f:
            f.write("config = " + json.dumps(setting, indent=4)
                    .replace('false', 'False').replace('true', 'True').replace('null', 'None'))
            f.write("\n\n")
            f.write(w3_calc_fn)
            f.write("\n\n")
            f.write(w4_calc_fn)
            f.write("\n")


def load_config_file(path: str) -> Tuple[Dict, Optional[Callable], Optional[Callable]]:
    config = None

    if os.path.exists(f"{path}"):
        import sys
        sys.path.append(f"{os.path.dirname(path)}")
        from config_model_env import config as c
        config = c
        try:
            from config_model_env import w3_calc
            from config_model_env import w4_calc

        except Exception as e:
            print("No w3 and w4 function found in config file.\
                    make sure their named w3_calc and w4_calc.")
            w3_calc = None
            w4_calc = None

        sys.path.pop()

    else:
        print(f"config file doesn't exist: {path}")

    return config, w3_calc, w4_calc


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

    config, w3_calc, w4_calc = load_config_file(path)

    if config and config.get('model', None):
        algorithm = algorithm or config['model'].get('algorithm', None)

    if algorithm is None:
        raise Exception("RL algorithm is not defined.\
            it should be given by CLI argument or exists in config file")

    if algorithm == 'dqn':
        model = DQN.load(f"{path}/model")
        print(f"DQN model is loaded from {path}/model.zip")
        if os.path.exists(f"{path}/replay_buffer.pkl"):
            model.load_replay_buffer(f"{path}/replay_buffer")
            print(f"DQN replay buffer is loaded from {path}/replay_buffer.pkl")

    elif algorithm == 'ppo':
        model = PPO.load(f"{path}/model")
        print(f"PPO model is loaded from {path}/model.zip")

    elif algorithm == "sac":
        model = SAC.load(f"{path}/model")
        print(f"SAC model is loaded from {path}/model.zip")
        if os.path.exists(f"{path}/replay_buffer.pkl"):
            model.load_replay_buffer(f"{path}/replay_buffer")
            print(f"SAC replay buffer is loaded from {path}/replay_buffer.pkl")

    return model, config, w3_calc, w4_calc


def setup():
    args = arg_parser()
