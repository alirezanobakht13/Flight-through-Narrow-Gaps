config = {
    "environment":{
        "target_x_movement_range": 2,
        "target_y_movement_range": 5,
        "target_z_movement_range": 2,
        "max_distance":40,
        "target_init_x":15,
        "target_init_z":-5,
        "distance_coefficient": 0.8
    }
}

def w4_calc(distance):
    return 1/(distance + 0.01)