# Aggressive drone flight through narrow tilted gaps

## Airsim

## Agent

### setup

Create conda environment

```powershell
conda env create -f environment.yml
```

then activate the environment and install airsim_gym module

```
conda activate airsim
pip install -e .
```

now do initial check py running

```
python .\reach_target_continuous_baselines.py -h
```

## TODOs

- [x] finish setup in utils.py
- [ ] check correction of utils.py
- [ ] add loggers and prints
- [ ] eval need criteria for success (in `if args.mode == 'eval'`)
