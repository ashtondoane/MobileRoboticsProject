import matplotlib.pyplot as plt
import numpy as np

# Ts=0.1

# State covariance
# q_x=2.3
# q_y=7
# q_yaw=0.0019
# q_v = 8.34
# q_w = 0.0037
q_x=(0.78+13.39)/2
q_y=7
q_yaw=0.00016
q_v = 2.29
q_w = 0.0008

#Should be squared values as it is a covariance matrix and not a variance matrix ???

Q = np.array([
    [q_x**2, 0, 0, 0, 0], 
    [0, q_y**2, 0, 0, 0], 
    [0, 0, q_yaw**2, 0, 0], 
    [0, 0, 0, q_v**2, 0], 
    [0, 0, 0, 0, q_w**2]
])

# process_noise=np.random.multivariate_normal(mean=np.zeros(Q.shape[0]), cov=Q)
process_noise=np.zeros(Q.shape[0])

# Observation covariance
# [TODO] see to calculate the values of r_x,r_y,r_yaw from the camera (or try and error)
# r_x=1
# r_y=1
# r_yaw=0.05
# r_v = 8.34
# r_w = 0.0037
r_x=1
r_y=1
r_yaw=1
r_v = 2.29
r_w = 0.0008

# Assuming r_x, r_y, r_yaw, r_v, and r_w are standard deviations
R_cam = np.array([
    [r_x**2, 0, 0, 0, 0],
    [0, r_y**2, 0, 0, 0],
    [0, 0, r_yaw**2, 0, 0],
    [0, 0, 0, r_v**2, 0],
    [0, 0, 0, 0, r_w**2]
])

R_odometry = np.array([
    [r_v**2, 0],
    [0, r_w**2]
])

# sensor_noise_cam=np.random.multivariate_normal(mean=np.zeros(R_cam.shape[0]), cov=R_cam)
# sensor_noise_odo=np.random.multivariate_normal(mean=np.zeros(R_odometry.shape[0]), cov=R_odometry)
sensor_noise_cam=np.zeros(R_cam.shape[0])
sensor_noise_odo=np.zeros(R_odometry.shape[0])

def get_state_transition_matrix(state, Ts):
    x, y, yaw, v, omega = state
    A = np.array([
        [1, 0, -v * np.sin(yaw) * Ts, np.cos(yaw) * Ts, 0],
        [0, 1, v * np.cos(yaw) * Ts, np.sin(yaw) * Ts, 0],
        [0, 0, 1, 0, Ts],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    return A

def get_control_matrix(yaw, Ts):
    B = np.array([[np.cos(yaw) * Ts, 0],
                  [np.sin(yaw) * Ts, 0],
                  [0, Ts],
                  [1, 0],
                  [0, 1]])
    return B

def ekf(state_est_prev, control_vect_prev, P_prev, obs_camera, obs_odometry, camera_state, Ts):
    """
    Estimates the current state using input sensor data and the previous state

    param state_est_prev: previous state a posteriori estimation (numpy array of shape (5,)) = [x, y, yaw, v, omega]
    param control_vect_prev: previous velocity vector (numpy array of shape (2,)) = [v, omega]
    param P_prev: previous state a posteriori covariance (numpy array of shape (5, 5))
    param obs_camera: observation vector from the camera (numpy array of shape (3,)) = [x, y, yaw]
    param obs_odometry: observation vector from odometry (numpy array of shape (2,)) = [v, omega]
    param camera_state: boolean, True if camera available, False otherwise
    param Ts: sampling time

    return state_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance
    """

    # Predict the state estimate
    yaw = state_est_prev[2]
    A = get_state_transition_matrix(state_est_prev, Ts)
    B = get_control_matrix(yaw, Ts)

    # Uncomment to introduce random process noise for realistic scenarios
    # process_noise = np.random.multivariate_normal(mean=np.zeros(Q.shape[0]), cov=Q)
    process_noise = np.zeros(Q.shape[0])

    state_pred = A @ state_est_prev + B @ control_vect_prev + process_noise

    # Predict the state covariance P_pred
    P_pred = A @ P_prev @ A.T + Q

    # Innovation / measurement residual
    if camera_state:
        R = R_cam
        sensor_noise = sensor_noise_cam
        obs_vect = np.concatenate((obs_camera, obs_odometry))
        H = np.eye(5)
    else:
        R = R_odometry
        sensor_noise = sensor_noise_odo
        obs_vect = obs_odometry
        H = np.array([[0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])

    i = obs_vect - (H @ state_pred + sensor_noise)

    # Measurement prediction covariance
    S = H @ P_pred @ H.T + R

    # Kalman gain
    K = P_pred @ H.T @ np.linalg.pinv(S)

    # Update state estimate
    state_est = state_pred + K @ i

    # Update covariance estimate (using numerically stable form)
    P_est = (np.eye(P_pred.shape[0]) - K @ H) @ P_pred

    return state_est, P_est