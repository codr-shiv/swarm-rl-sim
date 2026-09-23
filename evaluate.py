import os
import sys
import time
import cv2
import numpy as np

# Add root project dir to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from envs.frontier_env import MultiRobotFrontierEnv

def render_env(obs):
    """Visualizes the grid map and robots."""
    grid = obs['global_map']
    h, w = grid.shape
    
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Draw Map
    img[grid == -1] = [100, 100, 100]  # Gray for unknown
    img[grid == 0] = [255, 255, 255]   # White for free
    img[grid == 100] = [0, 0, 0]         # Black for obstacle
    
    # Draw Frontiers
    frontiers = obs['frontiers']
    num_valid = obs['num_valid_frontiers'][0]
    for i in range(num_valid):
        fx, fy = frontiers[i]
        cv2.circle(img, (fx, fy), 0, (0, 0, 255), -1) # Red dots for frontiers
        
    # Draw Robots
    r1, r2 = obs['robot_poses']
    cv2.circle(img, (r1[0], r1[1]), 1, (255, 0, 0), -1) # Blue robot
    cv2.circle(img, (r2[0], r2[1]), 1, (0, 255, 0), -1) # Green robot
    
    # Scale up for visibility
    img_scaled = cv2.resize(img, (600, 600), interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow("Multi-Robot Exploration RL", img_scaled)
    cv2.waitKey(100) # 10 FPS

if __name__ == "__main__":
    env = MultiRobotFrontierEnv()
    
    # Check if model exists
    if not os.path.exists("frontier_policy.zip"):
        print("Model 'frontier_policy.zip' not found. Please run train.py first.")
        sys.exit(1)
        
    print("Loading model...")
    model = PPO.load("frontier_policy")
    
    print("Starting evaluation episode...")
    obs, _ = env.reset()
    done = False
    
    total_reward = 0
    while not done:
        render_env(obs)
        
        # RL Agent predicts the best action
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
    print(f"Episode finished! Total Reward: {total_reward}")
    cv2.destroyAllWindows()
