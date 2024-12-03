import matplotlib.pyplot as plt
import numpy as np


def ekf(state_est_prev, control_vect_prev, P_prev, obs_camera, obs_odometry, camera_state, Ts):
    """
    Estimates the current state using input sensor data and the previous state
    
    param state_est_prev: previous state a posteriori estimation = [x, y, yaw, v, omega]
    param control_vect_prev: previous control vector = [v, omega]
    param P_prev: previous state a posteriori covariance
    param obs_camera: observation vector from the camera = [x, y, yaw]
    param obs_odometry: observation vector from the odometry = [v, omega]
    param camera_state: boolean, True if camera available, False otherwise
    param Ts: time step 

    return state_est: new a posteriori state estimation
    return P_est: new a posteriori state covariance
    """

    q_x=2
    q_y=0.3
    q_yaw=0.001
    q_v = 5
    q_w = 0.003

    Q = np.array([
        [q_x, 0, 0, 0, 0], 
        [0, q_y, 0, 0, 0], 
        [0, 0, q_yaw, 0, 0], 
        [0, 0, 0, q_v, 0], 
        [0, 0, 0, 0, q_w]
    ])

    r_p=1
    r_yaw=0.05
    r_v = 5
    r_w = 0.003

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


    # state_pred = A @ state_est_prev + B @ control_vect_prev 
    x, y, yaw, v, omega = state_est_prev
    v_cmd, omega_cmd = control_vect_prev
    
    # Prediction Step
    x_pred = x + Ts * v * np.cos(yaw)
    y_pred = y + Ts * v * np.sin(yaw)
    yaw_pred = yaw + Ts * omega
    v_pred = v_cmd
    omega_pred = omega_cmd
    state_pred = np.array([x_pred, y_pred, yaw_pred, v_pred, omega_pred])
    
    # Predict the state covariance P_pred
    # v=control_vect_prev[0]
    G = np.array([[1,0,-np.sin(yaw)*Ts*v,np.cos(yaw)*Ts,0],
                  [0,1,np.cos(yaw)*Ts*v, np.sin(yaw)*Ts,0],
                  [0,0,1,0,Ts],
                  [0,0,0,1,0],
                  [0,0,0,0,1]]) 
    
    P_pred =  G @ (P_prev @ G.T) + Q


    # inovation / measurement residual    
    if camera_state:
        R = R_cam
        obs_vect=np.concatenate((obs_camera,obs_odometry))
        H=np.eye(5)
    else:
        R = R_odometry
        obs_vect=obs_odometry
        H=np.array([[0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 1]])

    i = obs_vect - (H @ state_pred)
    
    # measurement prediction covariance
    S = H @ P_pred @ H.T + R

    # Kalman gain
    K = P_pred @ H.T @ np.linalg.pinv(S)

    # Update state estimate
    state_est = state_pred + K @ i
    # Update covariance estimate
    P_est = P_pred - K @ (H @ P_pred)

    return state_est, P_est
