import numpy as np

def bresenham_line(x0, y0, x1, y1):
    """Bresenham's Line Algorithm. Returns a list of (x,y) points."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
            
    points.append((x, y))
    return points

def compute_fov(robot_map, ground_truth, rx, ry, max_range=20):
    """
    Simulates a 360 LiDAR scan and updates the robot's belief map.
    robot_map: -1 = unknown, 0 = free, 1 = obstacle
    ground_truth: 0 = free, 1 = obstacle
    """
    height, width = ground_truth.shape
    num_rays = int(max_range * 2 * np.pi) # dense enough
    
    for i in range(num_rays):
        angle = (i / num_rays) * 2 * np.pi
        target_x = int(rx + max_range * np.cos(angle))
        target_y = int(ry + max_range * np.sin(angle))
        
        # Clamp target to grid bounds
        target_x = max(0, min(width - 1, target_x))
        target_y = max(0, min(height - 1, target_y))
        
        line = bresenham_line(rx, ry, target_x, target_y)
        
        for (x, y) in line:
            if x < 0 or x >= width or y < 0 or y >= height:
                break
                
            cell = ground_truth[y, x]
            robot_map[y, x] = cell
            
            if cell == 100:
                break # Hit an obstacle, stop ray
