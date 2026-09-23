import numpy as np
import random
import cv2

def generate_random_grid(size=64, num_obstacles_min=8, num_obstacles_max=15, min_clearance=4):
    """
    Generates a 2D numpy array representing the map.
    0 = Free space, 1 = Obstacle
    Returns:
        grid: np.array (size, size)
        spawns: list of (x, y) tuples for 2 robots
    """
    grid = np.zeros((size, size), dtype=np.uint8)
    
    # Boundary walls
    grid[0, :] = 100
    grid[-1, :] = 100
    grid[:, 0] = 100
    grid[:, -1] = 100
    
    num_obstacles = random.randint(num_obstacles_min, num_obstacles_max)
    
    for _ in range(num_obstacles):
        is_box = random.choice([True, False])
        if is_box:
            w = random.randint(3, 8)
            h = random.randint(3, 8)
            x = random.randint(1, size - w - 1)
            y = random.randint(1, size - h - 1)
            grid[y:y+h, x:x+w] = 100
        else:
            r = random.randint(2, 5)
            x = random.randint(r+1, size - r - 2)
            y = random.randint(r+1, size - r - 2)
            cv2.circle(grid, (x, y), r, 100, -1)
            
    # Find valid spawns (at least `min_clearance` away from any obstacle)
    dist_transform = cv2.distanceTransform((grid != 100).astype(np.uint8), cv2.DIST_L2, 5)
    valid_ys, valid_xs = np.where(dist_transform >= min_clearance)
    
    spawns = []
    if len(valid_xs) >= 2:
        # Pick two spawns far away from each other if possible
        idx1 = random.randint(0, len(valid_xs) - 1)
        s1 = (valid_xs[idx1], valid_ys[idx1])
        spawns.append(s1)
        
        best_dist = 0
        best_s2 = None
        for _ in range(10):
            idx2 = random.randint(0, len(valid_xs) - 1)
            s2 = (valid_xs[idx2], valid_ys[idx2])
            dist = np.hypot(s1[0] - s2[0], s1[1] - s2[1])
            if dist > best_dist:
                best_dist = dist
                best_s2 = s2
        
        spawns.append(best_s2)
    else:
        # Fallback (should rarely happen on an open map)
        spawns = [(5, 5), (size-5, size-5)]
        
    return grid, spawns
