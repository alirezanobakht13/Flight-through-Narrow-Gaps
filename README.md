# Aggressive drone flight through narrow tilted gaps

## Setup

### Simulator (AirSim)

1. Install **AirSim 1.6.0** with **Unreal Engine 4.27** on your machine based on its documentation [https://microsoft.github.io/AirSim/](https://microsoft.github.io/AirSim/) (you can use different version of each, but you may face unpredicted compatibility issues).
2. Add a gap (a square with a hole inside it). one similar asset is in *Unreal* directory (`Box_Brush3_StaticMesh_new2.uasset`).
3. set appropriate scale for the object.
4. make sure collision of object is enabled and the drone can pass through its hole.
5. change name of the object to `myobject` (it should be its ID Name). it is default name of the object. you can name it whatever you want but you should change it in config file too.
6. copy `setting.json` from *Unreal* directory to your home directory where AirSim pick the settings (for example in windows it's `C:\Users\<UserName>\Documents\AirSim`).

### Agent

Create conda environment

```powershell
conda env create -f environment.yml
```

then activate the environment and install airsim_gym module

```shell
conda activate airsim
pip install -e .
```

now do initial check py running

```shell
python ./agent/main.py -h
```

## How to run

First open Unreal project and click play button. after drone appeared in Unreal simulation panel, run following command to train or test (conda environment should be activated).

### Training

```shell
cd agent
python main.py train
```

### Test

```shell
cd agent
python main.py eval -l <path_to_specific_model_directory>
```

## Configs and Parameters

There are three types of parameters you can set. parameters for:

- Environment
- Model
- Simulation

for the first two types, there are five way of setting individual parameter prioritized in the following order:

1. CLI argument (like `python main.py train --learning_rate 0.001`)
2. given config file using command like `python main.py train --config <path_to_config_file>`
3. config file loaded when loading a model (when a model is saved, its corresponding config file will be saved too).
4. default config file `default_config.py`.
5. default value of model or environment itself will be selected. for some parameter there is no default value, so it will raise an error.

and for the last one you have to change values inside `setting.json` in AirSim setting picking directory.

### Environment Parameters

- `id`: name of the environment (let it be its default value).
- `ip_address`: IP of running simulator.
- `port`: port of running simulator.
- `target_init_x`: initial position of the gap in the **x** axis.
- `target_init_y`
- `target_init_z`
- `target_x_movement_range`: range of gap displacement with respect to initial position in **x** axis. gap position in x axis will be randomly chosen from `[target_init_x - target_x_movement_range, target_init_x + target_x_movement_range]`.
- `target_y_movement_range`
- `target_z_movement_range`
- `target_yaw_offset`: initial yaw of the gap.
- `target_pitch_offset`
- `target_roll_offset`
- `target_yaw_range`: range of changes in gap yaw. similar to `target_x_movement_range`.
- `target_pitch_range`
- `target_roll_range`
- `max_distance` maximum distance from center of the gap which drone can go without terminating the episode.
- `max_timestep`: maximum timestep of each episode.
- `time_or_distance_limit_passed_reward`: reward of time or distance limit. should be negative.
- `accident_reward`: reward of accident. it will be negated in program, so the value should be positive.
- `success_reward`: reward of successfully passing through the gap.
- `distance_coefficient`: weight of velocity reward.

There are two other parameters for the environment which only can be given through config files. this parameters are function and should be declared after `config` dictionary

- `w3_calc(distance)`: weight of *safety angle* reward which is function of distance between drone and center of gap. it should be something like:

```python
def w3_calc(distance):
    return 1/(distance + 0.001)
```

- `w4_calc(distance)`: weight of *safety margin* reward which is function of distance between drone and gap plane. it also should be something like function declared above.

### Model Parameters

- `algorithm`: sac or ppo.
- `gamma`: discount factor $\gamma$.
- `learning_rate`
- `batch_size`
- `gradient_steps`: number of neural network updates in each step.
- `policy_kwargs`: defines neural network architecture. see `default_config.py` for an example and more detail.
- `tensorboard_log`: file to save tensorboard logs.

### Simulation Parameters

check [https://microsoft.github.io/AirSim/settings/](https://microsoft.github.io/AirSim/settings/) for more detail.
