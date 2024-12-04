import numpy as np 
from scipy.ndimage import distance_transform_edt

def check_sides(position, dist_map, max_visibility=50, step=5):
    """
    Dynamically increase the visibility range to check for walls in left and right directions.
    Stops and returns the decision as soon as a wall is encountered on either side.
    
    Args:
        position (tuple): (x, y, theta) position and orientation of the robot.
        dist_map (2D numpy array): Precomputed distance transform of the global map.
        max_visibility (int): Maximum range to check.
        step (int): Step size to increment the visibility range.
    
    Returns:
        decision (str): 'left' or 'right' depending on the clearer direction.
        left_distance (float): Distance at which the wall was encountered on the left.
        right_distance (float): Distance at which the wall was encountered on the right.
    """
    x, y, theta = position
    direction_vector = np.array([np.cos(theta), np.sin(theta)])
    left_normal = np.array([-direction_vector[1], direction_vector[0]])  # 90° counterclockwise
    right_normal = np.array([direction_vector[1], -direction_vector[0]])  # 90° clockwise

    visibility_range = 0
    left_distance = right_distance = max_visibility  # Default to max visibility if no wall is encountered

    while visibility_range < max_visibility:
        visibility_range += step

        # Project to left and right positions
        left_pos = (int(round(x + visibility_range * left_normal[0])),
                    int(round(y + visibility_range * left_normal[1])))
        right_pos = (int(round(x + visibility_range * right_normal[0])),
                     int(round(y + visibility_range * right_normal[1])))

        # Clamp positions to map bounds
        left_pos = (min(max(left_pos[0], 0), dist_map.shape[1] - 1),
                    min(max(left_pos[1], 0), dist_map.shape[0] - 1))
        right_pos = (min(max(right_pos[0], 0), dist_map.shape[1] - 1),
                     min(max(right_pos[1], 0), dist_map.shape[0] - 1))

        # Get distances from the distance transform
        left_dist = dist_map[left_pos[1], left_pos[0]]
        right_dist = dist_map[right_pos[1], right_pos[0]]

        print(f"Visibility Range: {visibility_range}")
        print(f"Left Position: {left_pos}, Left Distance: {left_dist}")
        print(f"Right Position: {right_pos}, Right Distance: {right_dist}")

        # If a wall is encountered in either direction, stop and decide
        if left_dist < visibility_range and right_dist >= visibility_range:
            return "right", left_dist, right_dist
        if right_dist < visibility_range and left_dist >= visibility_range:
            return "left", left_dist, right_dist

        # If both sides encounter walls simultaneously, decide based on distances
        if left_dist < visibility_range and right_dist < visibility_range:
            return ("right" if right_dist > left_dist else "left"), left_dist, right_dist

    # If no walls encountered, choose based on max visibility distances
    return ("left" if left_distance > right_distance else "right"), left_distance, right_distance

