import gym
from gym import error, spaces, utils
from gym.utils import seeding

class AirsimGymReachTarget(gym.Env):
    metadata = {'render.modes':["rgb_array"]}

    def __init__(self) -> None:
        super().__init__()
    
    def step(self,action):
        NotImplementedError()

    def reset(self):
        NotImplementedError()

    def render(self,mode='rgb_array'):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()