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
        "accident_reward": 10,
        "success_reward": 6000,
        "time_or_distance_limit_passed_reward": -2,
        "distance_coefficient": 5
    },
    "model": {
        "algorithm": "sac",
        "gamma": 0.99,
        "learning_rate": None,
        "batch_size": 512,
        "tensorboard_log":'./tb_logs/',
        "policy_kwargs": {
            "net_arch": [
                {
                    "pi": [
                        256,
                        256,
                        256,
                        256,
                        256,
                        256,
                        256,
                        256
                    ],
                    "vf": [
                        256,
                        256,
                        256,
                        256,
                        256,
                        256,
                        256,
                        256
                    ]
                }
            ]
        }
    }
}
