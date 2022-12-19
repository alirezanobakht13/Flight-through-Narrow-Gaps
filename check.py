import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces

import numpy as np
import airsim as air

import airsim_gym
env = gym.make("airsim-gym-reach-target-continuous-v0")

print(env.get_config())