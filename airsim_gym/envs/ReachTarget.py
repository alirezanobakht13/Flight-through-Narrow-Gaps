import time
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import numpy as np
import airsim as air
from airsim.types import Pose,Quaternionr,Vector3r
import logging

class AirsimGymReachTarget(gym.Env):
    metadata = {'render.modes':["rgb_array"]}

    def __init__(self,
    ip_address="127.0.0.1",
    port=41451,
    movement_size=0.25,
    max_distance=20,
    target_init_x=0,
    target_init_y=10,
    target_init_z=-2,
    target_x_movement_range=2,
    target_y_movement_range=2,
    target_z_movement_range=2,
    target_name="myobject",
    max_timestep=10000,
    accident_reward=-10,
    success_reward= 30,
    time_or_distance_limit_passed_reward=-10,
    distance_coefficient = 1) -> None:
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
        self.max_distance=max_distance
        self.max_timestep = max_timestep
        self.timestep_count = 0
        self.accident_reward = accident_reward
        self.success_reward = success_reward
        self.time_or_distance_limit_passed_reward = time_or_distance_limit_passed_reward
        self.distance_coefficient = distance_coefficient

        self.pre_time = time.time()
        self.pre_distance = 0

        self.info = dict()
    
    def reset(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        self.drone.takeoffAsync().join()
        logging.info("drone tookoff")

        self.state['target_position'] = self._setup_target_position()
        
        self.state['time_passed_from_previous_step'] = np.zeros((1,),dtype=np.float32)

        self._compute_state()
        self.pre_distance = np.linalg.norm(self.state['position']
                                - self.state['target_position'])

        self.pre_time = time.time()
        self.timestep_count = 0

        return self.state
    
    def step(self,action):
        self.timestep_count += 1

        offset = self._interpret_action(action)

        vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity

        self.drone.moveByVelocityAsync(
            vel.x_val + offset[0],
            vel.y_val + offset[1],
            vel.z_val + offset[2],
            1
        )

        reward,done = self._reward_done()

        self._compute_state()

        self.state['time_passed_from_previous_step'] = np.array([time.time()-self.pre_time,],dtype=np.float32)

        self.pre_time = time.time()

        return self.state,reward,done,self.info

    def render(self,mode='rgb_array'):
        return self.state
    
    def close(self):
        self.drone.armDisarm(False)
        self.drone.reset()
        self.drone.enableApiControl(False)

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

        lv_x = k.linear_velocity.x_val
        lv_y = k.linear_velocity.y_val
        lv_z = k.linear_velocity.z_val
        self.state['linear_velocity'] = np.array([lv_x,lv_y,lv_z],dtype=np.float32)

        av_x = k.angular_velocity.x_val
        av_y = k.angular_velocity.y_val
        av_z = k.angular_velocity.z_val
        self.state['angular_velocity'] = np.array([av_x,av_y,av_z],dtype=np.float32)

    def _reward_done(self):

        distance = np.linalg.norm(self.state['position']
                                - self.state['target_position'])

        delta_dinstance = self.pre_distance - distance
        reward = self.distance_coefficient * delta_dinstance
        self.pre_distance = distance

        if self.drone.simGetCollisionInfo().has_collided:
            return self.accident_reward,True
        
        axis_distance = (self.state['position'] - self.state['target_position'])
        x_axis_distance = axis_distance[0]
        y_axis_distance = axis_distance[1]
        z_axis_distance = axis_distance[2]


        if y_axis_distance>=0: # Passed from the gate plane
            if abs(x_axis_distance) < 2.25 and abs(z_axis_distance)< 0.75: # passed through the gate
                return self.success_reward,True
            else:
                return reward,True
        
        if distance > self.max_distance or self.timestep_count > self.max_timestep:
            return self.time_or_distance_limit_passed_reward,True

        return reward,False

    def _interpret_action(self,action):
        offset = np.zeros((3,),dtype=np.float32)
        if action == 0:
            offset = np.array([0,0,-self.movement_size],dtype=np.float32)
        if action == 1:
            offset = np.array([0,0, self.movement_size],dtype=np.float32)
        if action == 2:
            offset = np.array([0, self.movement_size,0],dtype=np.float32)
        if action == 3:
            offset = np.array([0,-self.movement_size,0],dtype=np.float32)
        if action == 4:
            offset = np.array([-self.movement_size,0,0],dtype=np.float32)
        if action == 5:
            offset = np.array([ self.movement_size,0,0],dtype=np.float32)
        
        return offset