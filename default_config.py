config = {
    "environment": {
        "id": "airsim-gym-reach-target-continuous-v0",
        "target_x_movement_range": 0,
        "target_y_movement_range": 0,
        "target_z_movement_range": 0,
        "target_yaw_offset": 0,
        "target_pitch_offset": 0,
        "target_roll_offset": 0,
        "target_yaw_range": 0,
        "target_pitch_range": 0,
        "target_roll_range": 48,
        "accident_reward": 200,
        "success_reward": 6000,
        "time_or_distance_limit_passed_reward": -50,
        "distance_coefficient": 5
    },
    "model": {
        "algorithm": "sac",
        "gamma": 0.99,
        "learning_rate": None,
        "batch_size": 128,
        "tensorboard_log": './tb_logs/',
        "policy_kwargs": { # read https://stable-baselines3.readthedocs.io/en/v1.0/guide/custom_policy.html
            "net_arch": {
                "pi": [  # policy network
                    256,
                    256,
                    256,
                    256,
                    256,
                ],
                # "vf": [ # value network (for PPO)
                #     256,
                #     256,
                #     256,
                #     256,
                #     256,
                #     256,
                #     256,
                #     256
                # ],
                "qf": [  # critic network (for SAC, DDPG, TD3)
                    256,
                    256,
                    256,
                    256,
                    256,
                ]
            }
        }
    }
}


def w3_calc(distance):
    return 0

def w4_calc(distance):
    return 0