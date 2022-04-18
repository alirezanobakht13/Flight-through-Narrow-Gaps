from gym.envs.registration import register
from importlib_metadata import entry_points

register(
    id="airsim-gym-reach-target-v0",
    entry_point="airsim_gym.envs:AirsimGymReachTarget"
)

register(
    id="airsim-gym-reach-target-continuous-v0",
    entry_point="airsim_gym.envs:AirsimGymReachTargetContinuous"
)