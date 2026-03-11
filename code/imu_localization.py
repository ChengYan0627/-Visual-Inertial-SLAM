import numpy as np
from pr3_utils import load_data, axangle2pose, axangle2twist, twist2pose, inversePose, pose2adpose, visualize_trajectory_2d
import matplotlib.pyplot as plt

def ensure_3_by_T(x):
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got {x.shape}")
    if x.shape[0] == 3:
        return x
    elif x.shape[1] == 3:
        return x.T
    else:
        raise ValueError(f"Cannot interpret shape {x.shape} as (3,T)")

def imu_localization(v_t, w_t, timestamps):

    v_t = ensure_3_by_T(v_t)
    w_t = ensure_3_by_T(w_t)
    timestamps = np.asarray(timestamps).reshape(-1)
    num_steps = timestamps.shape[0]
    pose_trajectory = np.zeros((num_steps, 4, 4))

    # ==== EKF prior ====
    T_curr = np.eye(4) 
    sigma_curr = np.eye(6) * 1e-3 

    # noise
    noise_v = 1e-3
    noise_w = 1e-3
    W = np.diag([noise_v, noise_v, noise_v, noise_w, noise_w, noise_w])

    pose_trajectory[0] = T_curr

    # ==== EKF prediction ====
    for i in range(1, num_steps):

        dt = timestamps[i] - timestamps[i - 1]
        
        # twist
        v = v_t[:, i]
        w = w_t[:, i]

        twist = np.concatenate((v, w))

        # hat map
        hat_twist = axangle2twist(twist.reshape(1, 6))
        hat_twist_dt = hat_twist * dt

        # Exponential map
        exp_twist_dt = twist2pose(hat_twist_dt)[0]

        # Mean prediction
        T_curr = T_curr @ exp_twist_dt
        
        # Covariance prediction
        # Σ_{t+1|t} = Ft * Σ_{t|t} * Ft^T + noise
        # Ft (Adjoint Map)
        inv_twist_dt = inversePose(exp_twist_dt.reshape(1, 4, 4))
        F_t = pose2adpose(inv_twist_dt )[0] 
        Q_t = W * dt
        sigma_curr = F_t @ sigma_curr @ F_t.T + Q_t

        pose_trajectory[i] = T_curr

    return pose_trajectory, sigma_curr

def plot():

    filepath = "../data/dataset02/dataset02.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filepath)
    poses, sigmas = imu_localization(v_t, w_t, timestamps)

    fig, ax = visualize_trajectory_2d(poses, path_name="IMU Localization", show_ori=True)
    plt.show()

if __name__ == "__main__":
    plot()