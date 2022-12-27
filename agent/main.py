import time

import tqdm

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from common.utils import setup, save_model, logging
from common.callbacks import RewardCallback


current_time = time.strftime("%Y-%m-%d_%H-%M-%S")

model, env, envs, config, args = setup()

if args.mode == 'eval':

    episodes = args.get('episodes', None) or 50
    success = 0

    for e in tqdm(range(episodes)):
        obs = env.reset()

        while True:
            action, _state = model.predict(obs)
            obs, reward, done, info = env.step(action)

            if done:
                break

    print(f"number of success {success} | \
        total {episodes} | success rate {(success/episodes)*100}")

elif args.mode == 'train':
    callbacks = []
    eval_callback = EvalCallback(
        env,
        callback_on_new_best=None,
        n_eval_episodes=5,
        best_model_save_path=f"./models/best_models/continuous/{config['model']['algorithm']}/{current_time}",
        log_path=".",
        eval_freq=10000,
    )
    callbacks.append(eval_callback)
    callbacks.append(RewardCallback())

    checkpoint_callback = CheckpointCallback(save_freq=10000,
                                             save_path='./models/checkpoints/continuous',
                                             name_prefix=f"{config['model']['algorithm']}")
    callbacks.append(checkpoint_callback)

    kwargs = {}
    kwargs["callback"] = callbacks

    try:
        model.learn(
            total_timesteps=args.timesteps,
            tb_log_name=f"{config['model']['algorithm']}_run_" + current_time,
            **kwargs
        )

        save_model(
            model, f"models/train_finished/{config['model']['algorithm']}_{current_time}", config)

    except (KeyboardInterrupt, Exception) as e:
        logging.error("Exception occurred", exc_info=True)
        save_model(
            model, f"models/run_stopped/{config['model']['algorithm']}_{current_time}", config)
