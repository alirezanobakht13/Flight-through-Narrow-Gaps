import time
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import numpy as np
import airsim as air
from airsim.types import Pose,Quaternionr,Vector3r
import logging

class AirsimGymReachTargetContinuous(gym.Env):
    metadata = {'render.modes':["rgb_array"]}

    def __init__(self,
    ip_address="127.0.0.1",
    port=41451,
    movement_size=0.25,
    max_distance=20,
    target_init_x=10,
    target_init_y=0,
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
                dtype=np.float32
            ),
            
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

            'linear_acceleration':spaces.Box(
                low=np.array([-np.inf for _ in range(3)]),
                high=np.array([np.inf for _ in range(3)]),
                dtype=np.float32
            ),

            'angular_velocity':spaces.Box(
                low=np.array([-np.inf for _ in range(3)]),
                high=np.array([np.inf for _ in range(3)]),
                dtype=np.float32
            ),

            'angular_acceleration':spaces.Box(
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
            'linear_acceleration':np.zeros((3,),dtype=np.float32),
            'angular_velocity':np.zeros((3,),dtype=np.float32),
            'angular_acceleration':np.zeros((3,),dtype=np.float32),
            'target_position':np.zeros((3,),dtype=np.float32),
            'time_passed_from_previous_step':np.zeros((1,),dtype=np.float32)
        }

        self.action_space = spaces.Box(
            low=np.array([-1.0,-1.0,-1.0,0.0]),
            high=np.array([1.0,1.0,1.0,1.0]),
            dtype=np.float32
        )
        """roll_rate
        pitch_rate
        yaw_rate,
        throttle
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

        self.drone.moveByAngleRatesThrottleAsync(
            float(action[0]),
            float(action[1]),
            float(action[2]),
            float(action[3]),
            0.1
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

        la_x = k.linear_acceleration.x_val
        la_y = k.linear_acceleration.y_val
        la_z = k.linear_acceleration.z_val
        self.state['linear_acceleration'] = np.array([la_x,la_y,la_z],dtype=np.float32) 

        av_x = k.angular_velocity.x_val
        av_y = k.angular_velocity.y_val
        av_z = k.angular_velocity.z_val
        self.state['angular_velocity'] = np.array([av_x,av_y,av_z],dtype=np.float32)

        aa_x = k.angular_acceleration.x_val
        aa_y = k.angular_acceleration.y_val
        aa_z = k.angular_acceleration.z_val
        self.state['angular_acceleration'] = np.array([aa_x,aa_y,aa_z],dtype=np.float32)

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


        if x_axis_distance>=0: # Passed from the gate plane
            if abs(y_axis_distance) < 2.25 and abs(z_axis_distance)< 0.75: # passed through the gate
                return self.success_reward,True
            else:
                return self.accident_reward/2,True
        
        if distance > self.max_distance or self.timestep_count > self.max_timestep:
            return self.time_or_distance_limit_passed_reward,True

        return reward,False