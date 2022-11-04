import gym
import airsim_gym
import keyboard

env = gym.make(
    "airsim-gym-reach-target-v0",
    target_x_movement_range=0.1,
    target_y_movement_range=0.1,
    target_z_movement_range=0.1,
)

running = True
obs = env.reset()
reward,done,state = 0,0,0
print("initial state")
print(obs)

while running:
    if keyboard.is_pressed('esc'):
        running = False
        env.close()
        continue
    else:
        if keyboard.is_pressed('up_arrow'):
            obs,reward,done,state = env.step(0)
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
        elif keyboard.is_pressed('down_arrow'):
            obs,reward,done,state = env.step(1)
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
        elif keyboard.is_pressed('w'):
            obs,reward,done,state = env.step(2)
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
        elif keyboard.is_pressed('s'):
            obs,reward,done,state = env.step(3)
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
        elif keyboard.is_pressed('d'):
            obs,reward,done,state = env.step(4)
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
        elif keyboard.is_pressed('a'):
            obs,reward,done,state = env.step(5)
            print(f"obs: {obs}")
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"state: {state}")
