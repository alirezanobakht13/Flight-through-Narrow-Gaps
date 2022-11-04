import time
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym import spaces
import numpy as np
import airsim as air
from airsim.types import Pose,Quaternionr,Vector3r
from . import utils
import logging

class AirsimGymReachTargetContinuous(gym.Env):
    metadata = {'render.modes':["rgb_array"]}

    def __init__(self,
    ip_address="127.0.0.1",
    port=41451,
    movement_size=0.25,
    max_distance=30,
    target_init_x=25,
    target_init_y=0,
    target_init_z=-2,
    target_x_movement_range=2,
    target_y_movement_range=2,
    target_z_movement_range=2,
    target_yaw_offset=0,
    target_pitch_offset=0,
    target_roll_offset=0,
    target_yaw_range=1,
    target_pitch_range=1,
    target_roll_range=1,
    target_name="myobject",
    max_timestep=10000,
    accident_reward= 10,
    success_reward= 100,
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

            'target_orientation':spaces.Box(
                low=np.array([-np.inf for _ in range(4)]),
                high=np.array([np.inf for _ in range(4)]),
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
            'target_orientation':np.zeros((4,),dtype=np.float32),
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
        self.target_yaw_offset = target_yaw_offset
        self.target_pitch_offset = target_pitch_offset
        self.target_roll_offset = target_roll_offset
        self.target_yaw_range = target_yaw_range
        self.target_pitch_range = target_pitch_range
        self.target_roll_range = target_roll_range
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
        self.cur_time = time.time()
        self.pre_distance = 0

        self.info = dict()

    def reset(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        self.drone.takeoffAsync().join()
        logging.info("drone tookoff")

        self.state['target_position'], self.state['target_orientation'] = self._setup_target_position()
        
        self.state['time_passed_from_previous_step'] = np.zeros((1,),dtype=np.float32)

        self.info['success'] = False

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

        self.cur_time = time.time()

        reward,done = self._reward_done()

        self._compute_state()

        self.state['time_passed_from_previous_step'] = np.array([self.cur_time-self.pre_time,],dtype=np.float32)

        self.pre_time = self.cur_time

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
        
        yaw = np.random.uniform(
            self.target_yaw_offset-self.target_yaw_range,
            self.target_yaw_offset+self.target_yaw_range
        )

        pitch = np.random.uniform(
            self.target_pitch_offset-self.target_pitch_range,
            self.target_pitch_offset+self.target_pitch_range
        )

        roll = np.random.uniform(
            self.target_roll_offset-self.target_roll_range,
            self.target_roll_offset+self.target_roll_range
        )

        position = Vector3r(x,y,z)
        orientation = utils.to_quaternion(yaw, pitch, roll)

        self.info['Target_vector'] = dict()
        e1,e2,e3 = utils.to_orthogonal_vectors(orientation)
        self.info['Target_vector']['e1'] = e1
        self.info['Target_vector']['e2'] = e2
        self.info['Target_vector']['e3'] = e3



        if self.drone.simListSceneObjects(name_regex=self.target_name):
            self.drone.simSetObjectPose(self.target_name,Pose(position,orientation))
        else:
            raise Exception("There is no object with the given name in simulation environment")
        
        return (
            np.array([x,y,z],dtype=np.float32),
            np.array([orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val],
                    dtype=np.float32)
        )
    
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
        

        done = False

        distance = np.linalg.norm(self.state['position']
                                - self.state['target_position'])

        delta_dinstance = self.pre_distance - distance
        self.pre_distance = distance

        w,x,y,z = self.state['orientation'][0],\
                self.state['orientation'][1],\
                self.state['orientation'][2],\
                self.state['orientation'][3],

        u1,u2,u3 = utils.to_orthogonal_vectors(Quaternionr(x,y,z,w))
        e1 = self.info['Target_vector']['e1']
        e2 = self.info['Target_vector']['e2']
        e3 = self.info['Target_vector']['e3']

        sp = self.state['target_position'] - self.state['position']

        # ------------------------------ Facing the gap ------------------------------ #
        angle_u1_sp,r1 = utils.get_angle(u1,sp)
        self.info['angles'] = dict()
        self.info['angles']['u1_sp'] = angle_u1_sp

        # TODO Remember to add v0 and beta to this reward
        # --------------------------------- Velocity --------------------------------- #
        delta_time = self.cur_time-self.pre_time
        r2 = (delta_dinstance/delta_time)
        

        # ------------------------------- Safety Angle ------------------------------- #
        # angle_e2_u2,t1 = utils.get_angle(e2,utils.project_vector_on_plane(u2,e2,e3))
        angle_e2_u2,t1 = utils.get_angle(e2,u2)
        self.info['angles']['e2_u2'] = angle_e2_u2
        self.info['angles']['t1'] = t1
        # t1 = -t1**2
        angle_e3_u3,t2 = utils.get_angle(e3,u3)
        self.info['angles']['e3_u3'] = angle_e3_u3
        self.info['angles']['t2'] = t2
        # t2 = -t2**2
        # TODO add angle_e3_u3 too.
        # r3 = -(angle_e2_u2)/20
        r3 = 1/(1 - t1 + 0.002)

        # ------------------------------- Safety Margin ------------------------------ #
        margin = np.sqrt(np.dot(sp,e2)**2 + np.dot(sp,e3)**2)
        r4 = 1/((margin/10)+0.04)

        # -------------------------- Passing through the gap ------------------------- #
        theta,_ = utils.get_angle(sp,e1)
        self.info['angles']['sp_e1'] = theta
        if (theta < 0 or theta > 90) and abs(margin)<0.4:
            r5 = 1
            done = True
            self.info['success'] = True
            print(f"success final state: {self.state['position']}")
            print(f"theta: {theta}")
            print(f"sp: {sp}")
            print(f"margin: {margin}")
            print(f"u2: {u2}")
            print(f"e2: {e2}")
            print(f"angle_e2_u2: {angle_e2_u2}")
        else:
            r5 = 0
        
        if self.drone.simGetCollisionInfo().has_collided:
            done = True
            r6 = -1
        elif (theta < 0 or theta > 90) and abs(margin)>=0.4:
            done = True
            r6 = -0.5
            print(f"fail final state: {self.state['position']}")
            print(f"theta: {theta}")
            print(f"sp: {sp}")
            print(f"margin: {margin}")
            print(f"u2: {u2}")
            print(f"e2: {e2}")
            print(f"angle_e2_u2: {angle_e2_u2}")
        else:
            r6 = 0

        w1 = 0
        w2 = self.distance_coefficient
        w3 = self._w3_calc(distance)
        w4 = self._w4_calc(distance)
        w5 = self.success_reward
        w6 = self.accident_reward

        reward = w1*r1 + w2*r2 + w3*r3 + w4*r4 + w5*r5 + w6*r6

        self.info['distance'] = distance

        self.info['rewards'] = dict()
        self.info['rewards']['r1'] = r1
        self.info['rewards']['r2'] = r2
        self.info['rewards']['r3'] = r3
        self.info['rewards']['r4'] = r4
        self.info['rewards']['r5'] = r5
        self.info['rewards']['r6'] = r6

        self.info['rewards']['w1'] = w1
        self.info['rewards']['w2'] = w2
        self.info['rewards']['w3'] = w3
        self.info['rewards']['w4'] = w4
        self.info['rewards']['w5'] = w5
        self.info['rewards']['w6'] = w6

        self.info['rewards']['wr1'] = w1*r1
        self.info['rewards']['wr2'] = w2*r2
        self.info['rewards']['wr3'] = w3*r3
        self.info['rewards']['wr4'] = w4*r4
        self.info['rewards']['wr5'] = w5*r5
        self.info['rewards']['wr6'] = w6*r6

        self.info['done'] = done

        if distance > self.max_distance or self.timestep_count > self.max_timestep:
            return self.time_or_distance_limit_passed_reward,True
        
        return reward,done
    
    def _w3_calc(self, distance):
        # return self._normal_dist(distance, sd=3)
        return self._one_over_x_dist(distance)

    def _w4_calc(self, distance):
        # return self._normal_dist(distance, sd=1)
        return self._one_over_x_dist(distance)

    def _normal_dist(self, x, mean=0, sd=2):
        return (1/(np.sqrt(2*np.pi)*sd)) * np.exp(-0.5*((x-mean)/sd)**2)

    def _one_over_x_dist(self, distance, epsilon=0.1):
        return abs(1/(distance + epsilon))