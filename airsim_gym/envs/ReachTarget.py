import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import numpy as np
import airsim as air

class AirsimGymReachTarget(gym.Env):
    metadata = {'render.modes':["rgb_array"]}

    def __init__(self,
    ip_address="127.0.0.1",
    port=41451,
    movement_size=0.25,
    target_init_x=10,
    target_init_y=10,
    target_init_z=-10,
    target_x_movement_range=5,
    target_y_movement_range=5,
    target_z_movement_range=5,
    target_name="myobject") -> None:
        super().__init__()

        self.observation_space = spaces.Dict({
            'drone_position': spaces.Box(
                low=np.array([-np.inf for _ in range(3)]),
                high=np.array([np.inf for _ in range(3)]),
                dtype=np.float32),
            
            'orientation':spaces.Box(
                low=np.array([-np.inf for _ in range(4)]),
                high=np.array([np.inf for _ in range(4)]),
                dtype=np.float32
            ),

            'linear_velocity':spaces.Box(
                low=np.array([-np.inf for _ in range(3)]),
                high=np.array([np.inf for _ in range(3)]),
                dtype=np.float32
            ),

            'angular_velocity':spaces.Box(
                low=np.array([-np.inf for _ in range(3)]),
                high=np.array([np.inf for _ in range(3)]),
                dtype=np.float32
            ),

            'goal_position':spaces.Box(
                low=np.array([-np.inf for _ in range(3)]),
                high=np.array([np.inf for _ in range(3)]),
                dtype=np.float32
            ),

            'time_passed_from_previous_step': spaces.Box(low=-np.inf,high=np.inf,dtype=np.float32)
        })

        self.action_space = spaces.Discrete(6)
        """0:up
        1:down
        2:forward
        3:backward
        4:right
        5:left        
        """

        self.drone = air.MultirotorClient(ip=ip_address,port=port)

        self.target_init_x = target_init_x
        self.target_init_y = target_init_y
        self.target_init_z = target_init_z
        self.target_x_movement_range = target_x_movement_range
        self.target_y_movement_range = target_y_movement_range
        self.target_z_movement_range = target_z_movement_range
        self.target_name=target_name

        self.movement_size=movement_size
    
    def step(self,action):
        NotImplementedError()

    def reset(self):
        NotImplementedError()

    def render(self,mode='rgb_array'):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()