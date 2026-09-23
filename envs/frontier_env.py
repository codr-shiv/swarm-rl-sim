import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from rl_sim.core.map_generator import generate_random_grid
from rl_sim.core.raycaster import compute_fov

MAX_FRONTIERS = 20
GRID_SIZE = 64
MAX_STEPS = 500

class MultiRobotFrontierEnv(gym.Env):
    """
    A 2D Multi-Robot Exploration Environment.
    Both robots share the same map and are controlled simultaneously.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 10}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Action Space: [Robot1 Frontier Index, Robot2 Frontier Index]
        self.action_space = spaces.MultiDiscrete([MAX_FRONTIERS, MAX_FRONTIERS])
        
        # Observation Space
        self.observation_space = spaces.Dict({
            'global_map': spaces.Box(low=-1, high=100, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8),
            'robot_poses': spaces.Box(low=0, high=GRID_SIZE-1, shape=(2, 2), dtype=np.int32),
            'frontiers': spaces.Box(low=0, high=GRID_SIZE-1, shape=(MAX_FRONTIERS, 2), dtype=np.int32),
            'num_valid_frontiers': spaces.Box(low=0, high=MAX_FRONTIERS, shape=(1,), dtype=np.int32)
        })
        
    def _extract_frontiers(self):
        unknown_mask = np.where(self.belief_map == -1, 255, 0).astype(np.uint8)
        free_mask = np.where(self.belief_map == 0, 255, 0).astype(np.uint8)
        obstacle_mask = np.where(self.belief_map >= 50, 255, 0).astype(np.uint8)
        
        # Inflate obstacles (1 pixel is approx 0.15m in a scaled grid)
        obs_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_obstacles = cv2.dilate(obstacle_mask, obs_kernel, iterations=1)
        safe_free_mask = cv2.bitwise_and(free_mask, cv2.bitwise_not(dilated_obstacles))
        
        kernel = np.ones((3, 3), np.uint8)
        dilated_unknown = cv2.dilate(unknown_mask, kernel, iterations=1)
        frontier_mask = cv2.bitwise_and(safe_free_mask, dilated_unknown)
        
        num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(frontier_mask)
        
        frontiers = []
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 2: # Ignore tiny noise
                cx, cy = int(centroids[i][0]), int(centroids[i][1])
                frontiers.append([cx, cy])
                
        return frontiers

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Generate new ground truth map
        self.ground_truth, spawns = generate_random_grid(size=GRID_SIZE)
        
        # 2. Initialize Belief Map (all unknown)
        self.belief_map = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=np.int8)
        
        # 3. Set robot poses
        self.poses = np.array([list(spawns[0]), list(spawns[1])], dtype=np.int32)
        
        # 4. Initial Raycast
        compute_fov(self.belief_map, self.ground_truth, self.poses[0][0], self.poses[0][1])
        compute_fov(self.belief_map, self.ground_truth, self.poses[1][0], self.poses[1][1])
        
        self.current_step = 0
        self.explored_cells = np.sum(self.belief_map != -1)
        
        return self._get_obs(), {}

    def _get_obs(self):
        raw_frontiers = self._extract_frontiers()
        num_valid = min(len(raw_frontiers), MAX_FRONTIERS)
        
        padded_frontiers = np.zeros((MAX_FRONTIERS, 2), dtype=np.int32)
        if num_valid > 0:
            padded_frontiers[:num_valid] = raw_frontiers[:num_valid]
            
        return {
            'global_map': self.belief_map.copy(),
            'robot_poses': self.poses.copy(),
            'frontiers': padded_frontiers,
            'num_valid_frontiers': np.array([num_valid], dtype=np.int32)
        }

    def step(self, action):
        self.current_step += 1
        
        obs = self._get_obs()
        valid_frontiers = obs['frontiers']
        num_valid = obs['num_valid_frontiers'][0]
        
        reward = -0.1 # Step penalty
        
        # Move robots towards selected frontiers
        for i in range(2):
            f_idx = action[i]
            if f_idx < num_valid:
                target_x, target_y = valid_frontiers[f_idx]
                
                # Move 1 step towards target
                dx = target_x - self.poses[i][0]
                dy = target_y - self.poses[i][1]
                dist = np.hypot(dx, dy)
                
                if dist > 0:
                    step_x = int(round(self.poses[i][0] + (dx / dist)))
                    step_y = int(round(self.poses[i][1] + (dy / dist)))
                    
                    # Collision check with walls
                    if self.ground_truth[step_y, step_x] == 0:
                        self.poses[i][0] = step_x
                        self.poses[i][1] = step_y
        
        collision_count = 0
        # Inter-robot collision penalty
        if np.hypot(self.poses[0][0] - self.poses[1][0], self.poses[0][1] - self.poses[1][1]) < 2.0:
            reward -= 5.0
            collision_count = 1
            
        # Raycast again
        compute_fov(self.belief_map, self.ground_truth, self.poses[0][0], self.poses[0][1])
        compute_fov(self.belief_map, self.ground_truth, self.poses[1][0], self.poses[1][1])
        
        new_explored = np.sum(self.belief_map != -1)
        reward += (new_explored - self.explored_cells) * 1.0 # Reward for discovering new space
        self.explored_cells = new_explored
        
        # Check Done
        done = False
        if new_explored >= np.sum(self.ground_truth == 0) * 0.95: # 95% explored
            done = True
            reward += 100.0 # Completion bonus
            
        if self.current_step >= MAX_STEPS:
            done = True
            
        info = {
            'coverage_percentage': new_explored / np.sum(self.ground_truth == 0),
            'collision': collision_count
        }
            
        return self._get_obs(), reward, done, False, info
