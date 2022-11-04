import gym
import airsim_gym
import keyboard
import numpy as np


def show_data(obs,reward,done,info):
    # print(info['success'])

    # print(f"r3:{info['rewards']['r3']}")
    # print(f"d:{info['distance']}")
    # print(f"w3:{info['rewards']['w3']}")
    # print(f"wr3:{info['rewards']['wr3']}")

    # print(f"info['angles']['t1'] : {info['angles']['t1'] }")
    # print(f"info['angles']['t2'] : {info['angles']['t2'] }")

    if done:
        print(done)

    # print(f"obs: {obs}")
    # print(f"reward: {reward}")
    # print(f"done: {done}")
    # print(f"info: {info}")

env = gym.make(
    "airsim-gym-reach-target-continuous-v0",
    target_x_movement_range=0.1,
    target_y_movement_range=0.1,
    target_z_movement_range=0.1,
)

running = True
obs = env.reset()
reward,done,info = 0,0,0
print("initial info")
print(obs)

m = 0

while running:
    if keyboard.is_pressed('esc'):
        running = False
        env.close()
        print(m)
        continue
    else:
        if keyboard.is_pressed('up_arrow'):
            obs,reward,done,info = env.step(np.array([0,0,0,1]))
            show_data(obs,reward,done,info)
            
        elif keyboard.is_pressed('down_arrow'):
            obs,reward,done,info = env.step(np.array([0,0,0,0]))
            show_data(obs,reward,done,info)
            
        elif keyboard.is_pressed('w'):
            obs,reward,done,info = env.step(np.array([0,1,0,0.9]))
            show_data(obs,reward,done,info)
            
        elif keyboard.is_pressed('s'):
            obs,reward,done,info = env.step(np.array([0,-0.2,0,0.5]))
            show_data(obs,reward,done,info)

        elif keyboard.is_pressed('d'):
            obs,reward,done,info = env.step(np.array([0.2,0,0,0.5]))
            show_data(obs,reward,done,info)

        elif keyboard.is_pressed('a'):
            obs,reward,done,info = env.step(np.array([-0.2,0,0,0.5]))
            show_data(obs,reward,done,info)

        elif keyboard.is_pressed('q'):
            obs,reward,done,info = env.step(np.array([0,0,0.2,0.5]))
            show_data(obs,reward,done,info)

        elif keyboard.is_pressed('e'):
            obs,reward,done,info = env.step(np.array([0,0,-0.2,0.5]))
            show_data(obs,reward,done,info)
