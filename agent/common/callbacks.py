from stable_baselines3.common.callbacks import BaseCallback


class RewardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_reward_sum = 0
        self.episode_time_step = 0

    def _on_step(self) -> bool:
        """
        log cumulated reward, episode sum of reward, episode mean of reward and each reward with its weight in tensorboard
        """
        r = self.training_env.get_attr('info', 0)[0]['rewards']['R']
        self.episode_reward_sum += r
        self.episode_time_step += 1
        
        self.logger.record("my_records/total_reward", r)
        self.logger.record("my_records/wr1_facing_the_gap",
                           self.training_env.get_attr('info', 0)[0]['rewards']['wr1'])
        self.logger.record("my_records/wr2_velocity",
                           self.training_env.get_attr('info', 0)[0]['rewards']['wr2'])

        if self.training_env.get_attr('info', 0)[0]['done']:
            self.logger.record("my_records/wr3_safety_angle",
                               self.training_env.get_attr('info', 0)[0]['rewards']['wr3'])
            self.logger.record("my_records/wr4_safety_margin",
                               self.training_env.get_attr('info', 0)[0]['rewards']['wr4'])
            self.logger.record("my_records/wr5_passing",
                               self.training_env.get_attr('info', 0)[0]['rewards']['wr5'])
            self.logger.record("my_records/wr6_collision",
                               self.training_env.get_attr('info', 0)[0]['rewards']['wr6'])
            self.logger.record("my_records/episode_reward_sum",
                               self.episode_reward_sum)
            self.logger.record("my_records/episode_reward_mean",
                               self.episode_reward_sum/self.episode_time_step)
            
            self.episode_time_step = 0
            self.episode_reward_sum = 0
        return super()._on_step()
