# Multi-Robot Frontier Exploration: RL Simulation Architecture

This document outlines the core architecture of the `rl_sim` module, which serves as a highly specialized, standalone Gymnasium environment designed to train a centralized Reinforcement Learning policy for multi-robot frontier exploration. 

## Core Architecture

### 1. Environment Design & State Representation
The environment is built on `gymnasium` and uses a centralized controller that commands two robots simultaneously. 
To ensure the Stable Baselines 3 (SB3) model processes spatial geometry correctly (via `NatureCNN`), the observation space is structured as a **3-Channel Spatial Image** of size `160x160` with `uint8` values (0-255).

*   **Channel 0 (Occupancy Map):** Represents the merged belief map of the robots. Black (0) = Unknown, Gray (127) = Free Space, White (255) = Obstacles.
*   **Channel 1 (Robot Poses):** A black grid where the exact positions of the two robots are drawn as white pixels (255).
*   **Channel 2 (Frontier Poses):** A black grid where the centroids of all valid frontiers are drawn as white pixels (255).

This visual grounding prevents the neural network from having to mathematically learn coordinates, allowing it to "see" the relationship between robots, frontiers, and walls natively through convolutions.

### 2. Action Space
The agent outputs a `MultiDiscrete([20, 20])` array. Each integer corresponds to an index in the `obs['frontiers']` array. The model assigns Robot 1 to `Action[0]` and Robot 2 to `Action[1]`.

### 3. Physics & Gazebo Sim-to-Real Alignment
The simulation is heavily mathematically aligned with ROS 2 and Gazebo physics:
*   **Metric Scaling:** The grid is `160x160`, where `1 pixel = 0.05 meters`. This perfectly encompasses the `8.0m x 8.0m` random Gazebo arenas.
*   **LiDAR Simulation:** The raycaster accurately simulates the Turtlebot3 LDS-01 sensor with a `3.5m` range (`70` pixels).
*   **Nav2 Costmap Simulation:** In `reset()`, raw obstacles are inflated via `cv2.dilate` using a `5x5` kernel. This gives the 1-pixel mathematical robots a physical radius of `~0.105m`, preventing them from squeezing through narrow gaps or hugging walls, which would be rejected by Nav2 in real deployment.
*   **Movement & Sliding:** If a robot's path is blocked by the costmap, it attempts to "slide" along the X or Y axis. This mimics the obstacle avoidance behavior of the DWB local planner.

### 4. Reward Shaping
The reward structure is carefully scaled to balance exploration and safety:
*   **Exploration Reward:** Scaled proportionally so that exploring 100% of the free space yields exactly `+100.0` cumulative reward. 
*   **Completion Bonus:** `+50.0` when 95% of the map is explored.
*   **Time Penalty:** `-0.5` per step to encourage speed.
*   **Collision Penalty:** `-20.0` if robots come within `2.0` distance units of each other.
*   **Invalid Action Penalty:** `-1.0` if the agent chooses a frontier index that doesn't exist.

---

## Iterations

### Iteration 1 (Initial Configuration)
The first draft of the environment was a functional proof-of-concept but suffered from severe sim-to-real discrepancies and convergence failures:
*   **Resolution:** Used a `64x64` grid which did not match Gazebo metric sizes, requiring heavy coordinate transformation.
*   **State Space:** The map was a flat 2D matrix (`-1, 0, 100`). SB3 interpreted this as a numerical array, flattened it into a 4096-dimensional vector, and fed it into a basic MLP, resulting in terrible spatial awareness.
*   **Physics:** LiDAR raycasting was hardcoded to `20` pixels (1.0 meter), causing "tunnel vision." Movement was strictly Euclidean; if a robot grazed a wall, it became permanently stuck.
*   **Reward Scale:** Awarded `+1.0` per cell discovered. On a large map, this resulted in `+20,000` points. The `-5.0` collision penalty was mathematically ignored by the PPO optimizer because the magnitude of the discovery reward was too high.

### Iteration 2 (Sim-to-Real Alignment & Convergence Fixes)
To guarantee transferability to the ROS 2 stack, the following structural debugging and changes were executed:
1.  **Metric Correction:** Bumped `GRID_SIZE` to `160` and `MAX_STEPS` to `1000`. This ensures 1 pixel = 0.05m exactly for an 8x8m arena.
2.  **Sensor Alignment:** Increased the raycaster `max_range` to `70` pixels to perfectly match the Turtlebot3 `3.5m` hardware specification.
3.  **Neural Bottleneck Fix:** Transformed the observation space into a `(3, 160, 160)` 3-channel uint8 image. This forced SB3 to route the state through `NatureCNN`, massively improving spatial learning.
4.  **Nav2 Footprint Integration:** Implemented OpenCV obstacle dilation to simulate the Turtlebot3 radius (`0.105m`), and added X/Y axis wall-sliding to simulate the Nav2 local trajectory planner.
5.  **Reward Normalization:** Refactored the discovery reward to yield a maximum of `100.0` points for total map coverage. This normalized the Value function, allowing PPO to actually respect and optimize the increased `-20.0` collision penalties.
6.  **Perception Matching:** Matched the OpenCV extraction parameters in Python (`7x7` kernel, area `> 5`) exactly to the `frontier_coordinator.py` ROS 2 node so the AI sees identical frontiers in both simulation layers.
