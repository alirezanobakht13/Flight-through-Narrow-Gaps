from turtle import position
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import numpy as np
import airsim as air
from airsim.types import Pose,Quaternionr,Vector3r

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
            'position': spaces.Box(
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

            'target_position':spaces.Box(
                low=np.array([-np.inf for _ in range(3)]),
                high=np.array([np.inf for _ in range(3)]),
                dtype=np.float32
            ),

            'time_passed_from_previous_step': spaces.Box(low=-np.inf,high=np.inf,shape=(1,),dtype=np.float32)
        })

        self.state = {
            'position':np.zeros((3,),dtype=np.float32),
            'orientation':np.zeros((4,),dtype=np.float32),
            'linear_velocity':np.zeros((3,),dtype=np.float32),
            'angular_velocity':np.zeros((3,),dtype=np.float32),
            'target_position':np.zeros((3,),dtype=np.float32),
            'time_passed_from_previous_step':np.zeros((1,),dtype=np.float32)
        }

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
    
    def reset(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        self.state['target_position'] = self._setup_target_position()

        self.drone.takeoffAsync().join()

        self._compute_state()

        return self.state
    
    def step(self,action):
        NotImplementedError()

    def render(self,mode='rgb_array'):
        raise NotImplementedError()
    
    def close(self):
        raise NotImplementedError()

    def _setup_target_position(self):
        x = self.target_init_x + np.random.uniform(
            -self.target_x_movement_range,
            self.target_x_movement_range)

        y = self.target_init_y + np.random.uniform(
            -self.target_y_movement_range,
            self.target_y_movement_range)
        
        z = self.target_init_z + np.random.uniform(
            -self.target_z_movement_range,
            self.target_z_movement_range)

        if self.drone.simListSceneObjects(name_regex=self.target_name):
            position = Vector3r(x,y,z)
            orientation = Quaternionr(0,0,0,0)
            self.drone.simSetObjectPose(self.target_name,Pose(position,orientation))
        
        return np.array([x,y,z],dtype=np.float32)
    
    def _compute_state(self):
        k = self.drone.getMultirotorState().kinematics_estimated

        p_x = k.position.x_val
        p_y = k.position.y_val
        p_z = k.position.z_val
        self.state['position'] = np.array([p_x,p_y,p_z],dtype=np.float32)

        o_w = k.orientation.w_val
        o_x = k.orientation.x_val
        o_y = k.orientation.y_val
        o_z = k.orientation.z_val
        self.state['orientation'] = np.array([o_w,o_x,o_y,o_z],dtype=np.float32)

        # TODO Complete Other states
        # TODO Set time_passed_from_previous_step in step function
