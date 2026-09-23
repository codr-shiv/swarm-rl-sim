import os
import sys
import numpy as np

# Add root project dir to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from envs.frontier_env import MultiRobotFrontierEnv

class CustomTensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_collisions = 0
        self.episode_coverage = []

    def _on_step(self) -> bool:
        # Check if environment returned info dicts
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if 'collision' in info and info['collision'] > 0:
                    self.episode_collisions += info['collision']
                
                if 'coverage_percentage' in info:
                    self.episode_coverage.append(info['coverage_percentage'])
                    
                # Assuming "done" is in info or we can check terminal state
                if info.get('TimeLimit.truncated', False) or info.get('terminal_observation', None) is not None:
                    # Episode ended
                    self.logger.record("custom/episode_collisions", self.episode_collisions)
                    
                    if len(self.episode_coverage) > 0:
                        self.logger.record("custom/final_coverage", self.episode_coverage[-1])
                        
                    self.episode_collisions = 0
                    self.episode_coverage = []
        return True

def make_env():
    def _init():
        env = MultiRobotFrontierEnv()
        return env
    return _init

if __name__ == "__main__":
    # Use 4 parallel environments for faster training
    num_envs = 4
    env = SubprocVecEnv([make_env() for i in range(num_envs)])
    
    # Initialize PPO agent
    print("Initializing PPO MultiInputPolicy...")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        tensorboard_log="./ppo_frontier_tensorboard/"
    )
    
    # Train the agent
    total_timesteps = 10000000
    print(f"Starting training for {total_timesteps} timesteps...")
    
    custom_callback = CustomTensorboardCallback()
    model.learn(total_timesteps=total_timesteps, callback=custom_callback)
    
    # Save the trained policy
    model.save("frontier_policy")
    print("Training complete! Saved model to 'frontier_policy.zip'")
