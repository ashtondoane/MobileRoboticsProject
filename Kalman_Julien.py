import matplotlib.pyplot as plt
import numpy as np

# Ts=0.1

# State covariance

# q_p = q_x = q_y
q_p=0.04
q_v = 75.72

# VALUES TO BE DETERMINED
q_yaw=0.00016
q_w = 0.0008

#Should be squared values as it is a covariance matrix and not a variance matrix ???

Q = np.array([
    [q_p, 0, 0, 0, 0], 
    [0, q_p, 0, 0, 0], 
    [0, 0, q_yaw, 0, 0], 
    [0, 0, 0, q_v, 0], 
    [0, 0, 0, 0, q_w]
])

# process_noise=np.random.multivariate_normal(mean=np.zeros(Q.shape[0]), cov=Q)
process_noise=np.zeros(Q.shape[0])

# Observation covariance
# r_p = r_x = r_y

r_p = 0.25
r_v = 75.72

# VALUES TO BE DETERMINED
r_yaw=1
r_w = 0.0008

# Assuming r_x, r_y, r_yaw, r_v, and r_w are standard deviations
R_cam = np.array([
    [r_p, 0, 0, 0, 0],
    [0, r_p, 0, 0, 0],
    [0, 0, r_yaw, 0, 0],
    [0, 0, 0, r_v, 0],
    [0, 0, 0, 0, r_w]
])

R_odometry = np.array([
    [r_v, 0],
    [0, r_w]
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


def ekf_filter(x_est_prev, P_est_prev, Ts, measurement, Q, R, camera_state, obs_camera, obs_odometry):
    """
    Estimates the current state using input sensor data and the previous state
    
    param speed: measured linear speed (m/s)
    param angular_speed: measured angular speed (rad/s)
    param x_est_prev: previous state a posteriori estimation [x, y, yaw, v, w]
    param P_est_prev: previous state a posteriori covariance
    param Ts: sampling time interval (s)
    param measurement: measurement vector [x, y, yaw] from camera
    param Q: process noise covariance matrix
    param R: measurement noise covariance matrix
    
    return x_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance
    """

    # Define the state transition model A (Jacobian of the motion model w.r.t state)
    A = get_state_transition_matrix(x_est_prev, Ts)
    
    # Predict the next state (non-linear state update)
    x_est_a_priori = A @ x_est_prev

     # Predict the error covariance
    P_est_a_priori = A @ (P_est_prev @ A.T)
    P_est_a_priori = P_est_a_priori + Q if type(Q) != type(None) else P_est_a_priori
    
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

    i = obs_vect - (H @ x_est_a_priori + sensor_noise)

    # Measurement prediction covariance
    S = H @ P_est_a_priori @ H.T + R

    # Kalman gain
    K = P_est_a_priori @ H.T @ np.linalg.pinv(S)

    # Update the state with the measurement
    x_est = x_est_a_priori + K @ i

    # Update the error covariance
    P_est = P_est_a_priori - K @ (H @ P_est_a_priori)

    return x_est, P_est