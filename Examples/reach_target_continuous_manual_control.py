import gym
import airsim_gym
import keyboard
import numpy as np

env = gym.make(
    "airsim-gym-reach-target-continuous-v0",
    target_x_movement_range=0.1,
    target_y_movement_range=0.1,
    target_z_movement_range=0.1,
)

running = True
obs = env.reset()
reward,done,state = 0,0,0
print("initial state")
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
            obs,reward,done,state = env.step(np.array([0,0,0,1]))
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
            # m1 = np.linalg.norm(obs["linear_velocity"])
            # print(m1)
            # if m1>m:
            #     m = m1
        elif keyboard.is_pressed('down_arrow'):
            obs,reward,done,state = env.step(np.array([0,0,0,0]))
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
            # m1 = np.linalg.norm(obs["linear_velocity"])
            # print(m1)
            # if m1>m:
            #     m = m1
        elif keyboard.is_pressed('w'):
            obs,reward,done,state = env.step(np.array([0,0.2,0,0.9]))
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
            m1 = obs["linear_velocity"][1]
            print(m1)
            if m1>m:
                m = m1
        elif keyboard.is_pressed('s'):
            obs,reward,done,state = env.step(np.array([0,-0.2,0,0.5]))
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
        elif keyboard.is_pressed('d'):
            obs,reward,done,state = env.step(np.array([0.2,0,0,0.5]))
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
        elif keyboard.is_pressed('a'):
            obs,reward,done,state = env.step(np.array([-0.2,0,0,0.5]))
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
        elif keyboard.is_pressed('q'):
            obs,reward,done,state = env.step(np.array([0,0,0.2,0.5]))
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")

        elif keyboard.is_pressed('e'):
            obs,reward,done,state = env.step(np.array([0,0,-0.2,0.5]))
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
