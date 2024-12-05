import numpy as np
# Robot-specific constants
ROBOT_CENTER_TO_WHEEL = 47.5  # Half the wheelbase mm
ROBOT_WHEEL_RADIUS = 22    # Radius of the wheels mm
SPEED_THRESHOLD = 100         # Max Thymio wheel speed

# Normalize angle to (-pi, pi)
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# Astolfi Controller for Thymio
def astolfi_controller(state, goal, kr= 0.8 , ka=  1.4 , goal_tolerance=20):

    """ 
    Astolfi Controller Implementation

    Inputs 
        states : position of thymio :x , y and yaw [mm]
        goal : 2D corrdinates of goal [mm]
        Kr : has to be positive
        Ka : has to be negative 
        goal tolerance : threshold for saying if goal is reached in mm

    Outputs: 
        v : linear velocity of thymio in mm/s
        w : Angular velocity of thymio in rad/s
        reached : Boolean indicatinf if goal is reached 
     """
    #alpha doit etre plus grand que rho pour Julien 
    
    x = state[0] #en mm 
    y = state[1] #en mm 
    theta = state[2]
    x_goal, y_goal = goal
    dx = x_goal - x
    dy = -(y_goal - y)
    rho = np.sqrt(dx**2 + dy**2)  # Distance to goal

    # print ("Distance to goal = " , rho )

    # alpha = -normalize_angle(np.arctan2(dy, dx) - theta)
    alpha = (np.arctan2(dy, dx) - theta)
    #beta = normalize_angle(- theta - alpha)

    # Stop if the robot is close enough to the goal
    if rho < goal_tolerance:
        return 0.0, 0.0, True
    # Control laws
    v = kr * rho #kr > 0 
    #v = max(kr * rho, MIN_SPEED) 
    omega = ka * alpha 
    return v, omega, False


# Convert velocities to motor speeds
def compute_motor_speeds(v, omega):

    """ converts the control inputs of astolfi to thymio's units basically """

     # Convert speed thresholds for clipping
    max_v = SPEED_THRESHOLD * 0.4375  # Max linear velocity in mm/s
    max_omega = max_v / ROBOT_CENTER_TO_WHEEL  # Max angular velocity in rad/s

    v = np.clip(v, -max_v, max_v) #actul control velocity given to the thymio 
    omega = np.clip(omega, -max_omega, max_omega)

    right_wheel_mm_s = (v + omega * ROBOT_CENTER_TO_WHEEL)  # mm/s
    left_wheel_mm_s = (v - omega * ROBOT_CENTER_TO_WHEEL)  # mm/s

    thymio_speed_to_mms = 0.388
    right_wheel_enc = np.clip(right_wheel_mm_s / thymio_speed_to_mms, -SPEED_THRESHOLD, SPEED_THRESHOLD)
    left_wheel_enc = np.clip(left_wheel_mm_s / thymio_speed_to_mms, -SPEED_THRESHOLD, SPEED_THRESHOLD)

    # # Debugging outputs
    # print(f"Clipped control inputs: v = {v}, omega = {omega}")
    # print(f"left wheel control speed = {left_wheel_enc}, right wheel control speed = {right_wheel_enc}")

    return int(left_wheel_enc), int(right_wheel_enc) , v , omega  


def move_to_waypoint(state, waypoint, goal_tolerance=20):
    # Astolfi controller to compute control signals
    v, omega, reached = astolfi_controller(state, waypoint, goal_tolerance=goal_tolerance)
    if reached:
        return 0, 0, True , 0 ,0
    # Convert control signals to motor speeds
    left_speed, right_speed ,v_f , omega_f = compute_motor_speeds(v, omega)
    return left_speed, right_speed, False , v_f , omega_f  #in Thymio's units 

def segment_path(path, step=10):
    """
    Segments the A* path into waypoints for the motion controller.

    Parameters:
    - path: List of (row, column) points from the A* algorithm.
    - step: Number of points to skip between waypoints.

    Returns:
    - waypoints: List of (x, y) waypoints in meters.
    """
    waypoints = path[::step]  # Take every `step`-th point
    waypoints.append(path[-1])  # Ensure the last point (goal) is included
    return waypoints

def grid_to_world_coordinates(waypoints, map_resolution, origin=(0, 0)):
    """
    Converts waypoints from grid coordinates to world coordinates.

    Parameters:
    - waypoints: List of (row, col) waypoints in grid coordinates.
    - map_resolution: Size of each grid cell in meters.
    - origin: World coordinates of the grid's (0, 0).

    Returns:
    - world_waypoints: List of (x, y) waypoints in meters.
    """
    world_waypoints = [
        (origin[0] + w[1] * map_resolution, origin[1] + w[0] * map_resolution) for w in waypoints
    ]
    return world_waypoints

#Local Nav 

def avoid_obstacle(proximity_values , obstSpeedGain) : 

    prox_left, prox_left_front, prox_front, prox_right_front, prox_right = proximity_values

    #baseline speed
    spLeft = 50 
    spRight = 50

    if (abs(prox_left_front - prox_right_front) < 20 ) : #handling special case where the obstacle is just in front and thymio can't seem to make a decision  
        if prox_left_front >= prox_right_front:
            # More space on the right, turn right
            print("Turning right to avoid obstacle")
            spLeft += 40  # Boost left wheel speed to turn right
            spRight -= 40  # Slow down right wheel speed
        elif  prox_right_front >= prox_left_front:
            # More space on the left, turn left
            print("Turning left to avoid obstacle")
            spLeft -= 40  # Slow down left wheel speed
            spRight += 40  # Boost right wheedl speed
        else: 
            print("Turning right to avoid obstacle par defaut ")
            spLeft += 40  # Boost left wheel speed to turn right
            spRight -= 40  # Slow down right wheel speed
    else: 
        for i in range(5):
            spLeft += proximity_values[i] * obstSpeedGain[i] // 100
            spRight += proximity_values[i] * obstSpeedGain[4 - i] // 100

    return spLeft , spRight 