import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from envs.frontier_env import MultiRobotFrontierEnv

def evaluate_policy_headless(model_path="frontier_policy.zip", num_episodes=100):
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found.")
        return
        
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    env = MultiRobotFrontierEnv()
    
    total_rewards = []
    total_coverages = []
    total_collisions = []
    total_steps = []
    
    print(f"Evaluating for {num_episodes} episodes...")
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        ep_reward = 0
        ep_collisions = 0
        steps = 0
        final_coverage = 0
        
        while not done:
            # deterministic=True forces the policy to pick the best action without random exploration
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if info.get('collision', 0) > 0:
                ep_collisions += info['collision']
            final_coverage = info.get('coverage_percentage', 0)
            
        total_rewards.append(ep_reward)
        total_coverages.append(final_coverage)
        total_collisions.append(ep_collisions)
        total_steps.append(steps)
        
    print("\n=== FINAL RESULTS (Averaged over 100 episodes) ===")
    print(f"Average Reward:     {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Coverage:   {np.mean(total_coverages):.1%} ± {np.std(total_coverages):.1%}")
    print(f"Average Collisions: {np.mean(total_collisions):.2f} ± {np.std(total_collisions):.2f}")
    print(f"Average Steps:      {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    
    if np.mean(total_coverages) >= 0.90 and np.mean(total_collisions) <= 0.5:
        print("\nVerdict: Model is performing VERY WELL (High coverage, low collisions).")
    elif np.mean(total_coverages) >= 0.85:
        print("\nVerdict: Model is MEDIOCRE. Explores decently but likely collides too often.")
    else:
        print("\nVerdict: Model has NOT CONVERGED. It gets stuck or fails to explore.")

if __name__ == "__main__":
    evaluate_policy_headless()
