from typing import Union, Optional, Dict, Text
import json
import argparse

from stable_baselines3 import PPO,DQN,SAC

def arg_parser():
    """
    CLI arguments are defined here.
    """
    parser = argparse.ArgumentParser(description="baseline algorithms for training and evaluating on airsim_gym environment.\
                                            Note that you should have stable baseline3 installed \
                                            (use 'pip install stable-baselines3').")

    parser.add_argument(
        "mode",
        choices=['train','eval'],
        help="select training or evaluating"
    )

    parser.add_argument(
        "algorithm",
        choices=['ppo','sac'],
        help="which RL algorithm you want for train or evaluating"
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

    args = parser.parse_args()
    return args


def save_model(
    model: Union[PPO,DQN,SAC],
    path: Text,
    detail: Optional[Dict]
) -> None:
    """
    save the model to given path.
    detail could contain information about:
    - reward weights
    - network architecture
    - gamma
    - learning rate
    - ...
    """

    print(f"saving model to {path}")
    model.save(path)

    if isinstance(model,SAC) or isinstance(model,DQN):
        print(f"save model replay buffer to {path}_rep")
        model.save_replay_buffer(f"{path}_rep")
    
    if detail:
        print(f"save model detail to {path}_detail")
        with open(f"{path}_detail.json", "w") as f:
            json.dump(detail,f,indent=4)

