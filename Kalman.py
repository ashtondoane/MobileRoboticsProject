import matplotlib.pyplot as plt
import numpy as np


Ts=0.1

A = np.array([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,0,0], [0,0,0,0,0]])

# State covariance
q_x=2.3
q_y=7
q_yaw=0.0019
q_v = 8.34
q_w = 0.0037

Q = np.array([[q_x,0,0,0,0],[0,q_y,0,0,0],[0,0,q_yaw,0,0],[0,0,0,q_v,0],[0,0,0,0,q_w]])

# process_noise=np.random.multivariate_normal(mean=np.zeros(Q.shape[0]), cov=Q)
process_noise=np.zeros(Q.shape[0])

# Observation covariance
# [TODO] see to calculate the values of r_x,r_y,r_yaw from the camera (or try and error)
r_x=1
r_y=1
r_yaw=0.05
r_v = 8.34
r_w = 0.0037

R_cam = np.array([[r_x,0,0,0,0],[0,r_y,0,0,0],[0,0,r_yaw,0,0],[0,0,0,r_v,0],[0,0,0,0,r_w]])
R_odometry=np.array([[r_v,0],[0,r_w]])

# sensor_noise_cam=np.random.multivariate_normal(mean=np.zeros(R_cam.shape[0]), cov=R_cam)
# sensor_noise_odo=np.random.multivariate_normal(mean=np.zeros(R_odometry.shape[0]), cov=R_odometry)
sensor_noise_cam=np.zeros(R_cam.shape[0])
sensor_noise_odo=np.zeros(R_odometry.shape[0])


def ekf(state_est_prev, control_vect_prev, P_prev, obs_camera, obs_odometry, camera_state):
    """
    Estimates the current state using input sensor data and the previous state
    
    param state_est_prev: previous state a posteriori estimation = [x, y, yaw, v, omega]
    param control_vect_prev: previous velocity vector = [v, omega]
    param P_prev: previous state a posteriori covariance
    param obs_camera: observation vector = [x, y, yaw,]
    param obs_odometry: observation vector = [v, omega]
    param camera_state: boolean, True if camera available, False otherwise

    return state_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance
    """
    # Predict the state estimate
    yaw=state_est_prev[2]
    B = np.array([[np.cos(yaw)*Ts,0],[np.sin(yaw)*Ts,0],[0, Ts],[1,0],[0,1]])

    state_pred = A @ state_est_prev + B @ control_vect_prev + process_noise


    # Predict the state covariance P_pred
    v=control_vect_prev[0]
    G = np.array([[1,0,-np.sin(yaw)*Ts*v,np.cos(yaw)*Ts,0],[0,1,np.cos(yaw)*Ts*v, np.sin(yaw)*Ts,0],[0,0,1,0,Ts],[0,0,0,1,0],[0,0,0,0,1]]) 
    
    P_pred =  G @ P_prev @ G.T + Q


    # inovation / measurement residual    
    if camera_state:
        R = R_cam
        sensor_noise = sensor_noise_cam
        obs_vect=np.concatenate((obs_camera,obs_odometry))
        H=np.eye(5)
    else:
        R = R_odometry
        sensor_noise = sensor_noise_odo
        obs_vect=obs_odometry
        H=np.array([[0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])

    i = obs_vect - (H @ state_pred + sensor_noise)
    
    # measurement prediction covariance
    S = H @ P_pred @ H.T + R

    # Kalman gain
    K = P_pred @ H.T @ np.linalg.pinv(S)

    # Update state estimate
    state_est = state_pred + K @ i
    # Update covariance estimate
    P_est = P_pred - K @ H @ P_pred

    return state_est, P_est


def speed_convesion(r_speed,l_speed):
    thymio_speed_to_mms = 0.430 # value found in covariance_estimation

    #odometry 
    avg_thymio_speed = (r_speed + l_speed) / 2
    speed = avg_thymio_speed * thymio_speed_to_mms # [mm/s]
    return speed

def angular_vel_conversion(r_speed,l_speed):
    d = 95 # distance between the 2 wheels [mm]
    thymio_speed_to_mms = 0.430 # value found in covariance_estimation
    
    difference_speed = l_speed - r_speed
    omega = difference_speed * thymio_speed_to_mms / d # [rad/s]

    return omega