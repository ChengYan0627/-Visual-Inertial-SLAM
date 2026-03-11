import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from pr3_utils import load_data, axangle2twist, twist2pose, inversePose, pose2adpose, mat2euler

IMU_NOISE = 1e-8
CAMERA_NOISE = np.eye(4) * 10000
SIGMA_POSE = np.eye(6) * 1e-2

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
    
def circle_dot(s):
    x, y, z, w = s
    return np.array([
        [w, 0, 0,  0,  z, -y],
        [0, w, 0, -z,  0,  x],
        [0, 0, w,  y, -x,  0],
        [0, 0, 0,  0,  0,  0]
    ])

def initialize_landmark(z, T, Ks, leftcam_T_imu):
    f_u, c_u, f_v, c_v = Ks[0,0], Ks[0,2], Ks[1,1], Ks[1,2]
    fu_b = -Ks[2, 3] 
    disparity = z[0] - z[2]
    if disparity <= 2.0:
        return None
    Z = fu_b / disparity

    X = (z[0] - c_u) * Z / f_u
    Y = (z[1] - c_v) * Z / f_v
    P_cam = np.array([X, Y, Z, 1])
    m_world = T @ np.linalg.inv(leftcam_T_imu) @ P_cam
    return m_world[:3]

def jacobian_H(q, K_cam, cam_T_imu, T_inv, m_homo, P_T):
    x, y, z, _ = q
    dpi_dq = (1.0 / z) * np.array([
        [1, 0, -x/z, 0],
        [0, 1, -y/z, 0],
        [0, 0,    0, 0]  
    ])
    
    # Jacobian H_map (3x3)
    H_map = K_cam @ dpi_dq @ cam_T_imu @ T_inv @ P_T
    H_map = H_map[:2, :] # 2x3
    
    # Jacobian H_pose (3x6) 
    p = T_inv @ m_homo
    H_pose = K_cam @ dpi_dq @ cam_T_imu @ (-circle_dot(p))
    H_pose = H_pose[:2, :] # 2x6
    
    return H_map, H_pose

def visual_inertial_slam(v_t, w_t, timestamps, z_t, K_l, K_r, extL_T_imu, extR_T_imu):

    print("v_t shape:", v_t.shape)
    print("w_t shape:", w_t.shape)
    print("timestamps shape:", timestamps.shape)
    print("z_t shape:", z_t.shape)

    v_t = ensure_3_by_T(np.squeeze(v_t))
    w_t = ensure_3_by_T(np.squeeze(w_t))         
    timestamps = np.asarray(timestamps).reshape(-1)  

    _, features_M, total_t = z_t.shape
    
    T_curr = np.eye(4)
    sigma_pose = SIGMA_POSE
    pose_trajectory = np.zeros((total_t, 4, 4))
    pose_trajectory[0] = T_curr
    
    mu = np.zeros((3, features_M))
    sigma_map = sparse.lil_matrix((3 * features_M, 3 * features_M))
    sigma_map.setdiag(0.1)
    initialized = np.zeros(features_M, dtype=bool)
    
    # noise
    W = np.diag([IMU_NOISE, IMU_NOISE, IMU_NOISE, IMU_NOISE, IMU_NOISE, IMU_NOISE])

    V = CAMERA_NOISE        
    P_T = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]]) 
    
    baseline = np.linalg.norm(extL_T_imu[:3, 3] - extR_T_imu[:3, 3])
    Ks_approx = np.array([
        [K_l[0,0], 0, K_l[0,2], 0],
        [0, K_l[1,1], K_l[1,2], 0],
        [K_l[0,0], 0, K_l[0,2], -K_l[0,0] * baseline],
        [0, K_l[1,1], K_l[1,2], 0]
    ])

    for t in range(1, total_t):
        if t % 100 == 0: print(f"Running: {t} / {total_t}")
            
        # ====== Prediction step ======
        dt = float(timestamps[t] - timestamps[t - 1])

        v = v_t[:, t - 1]
        w = w_t[:, t - 1]

        twist = np.concatenate((v, w))
        
        # Pose Mean
        hat_twist_dt = axangle2twist(twist.reshape(1, 6)) * dt
        exp_twist_dt = twist2pose(hat_twist_dt)[0]
        
        T_curr = T_curr @ exp_twist_dt

        # Pose Sigma
        inv_twist_dt = inversePose(exp_twist_dt.reshape(1, 4, 4))
        F_t = pose2adpose(inv_twist_dt)[0] 
        sigma_pose = F_t @ sigma_pose @ F_t.T + (W * dt)
        
        # ====== Update step ======
        curr_T_inv = np.linalg.inv(T_curr) 
        
        for j in range(features_M):
            z = z_t[:, j, t]
            if z[0] == -1: 
                continue 
            
            # new landmark
            if not initialized[j]:
                init_m = initialize_landmark(z, T_curr, Ks_approx, extL_T_imu)
                if init_m is None: 
                    continue 

                mu[:, j] = init_m
                initialized[j] = True
                continue

            mu_homo = np.append(mu[:, j], 1)
            q_L = extL_T_imu @ curr_T_inv @ mu_homo
            q_R = extR_T_imu @ curr_T_inv @ mu_homo
            if q_L[2] <= 0 or q_R[2] <= 0: 
                continue
            
            # estimate landmark z_hat
            u_L = K_l[0,0] * (q_L[0] / q_L[2]) + K_l[0,2]
            v_L = K_l[1,1] * (q_L[1] / q_L[2]) + K_l[1,2]
            u_R = K_r[0,0] * (q_R[0] / q_R[2]) + K_r[0,2]
            v_R = K_r[1,1] * (q_R[1] / q_R[2]) + K_r[1,2]
            z_hat = np.array([u_L, v_L, u_R, v_R])
            
            # Jacobian H
            H_L_map, H_L_pose = jacobian_H(q_L, K_l, extL_T_imu, curr_T_inv, mu_homo, P_T)
            H_R_map, H_R_pose = jacobian_H(q_R, K_r, extR_T_imu, curr_T_inv, mu_homo, P_T)
            
            H_map_j = np.vstack((H_L_map, H_R_map))   # 4x3
            H_pose_j = np.vstack((H_L_pose, H_R_pose)) # 4x6
            
            # get sigma
            sigma_m_j = sigma_map[3*j:3*j+3, 3*j:3*j+3].toarray() 
            
            # K gain 
            S = H_map_j @ sigma_m_j @ H_map_j.T + H_pose_j @ sigma_pose @ H_pose_j.T + V
            S_inv = np.linalg.inv(S)
            
            K_gain_map = sigma_m_j @ H_map_j.T @ S_inv   # 3x4
            K_gain_pose = sigma_pose @ H_pose_j.T @ S_inv # 6x4
            
            innovation = z - z_hat
            
            # update map mean
            mu[:, j] += K_gain_map @ innovation
            
            # update pose mean
            delta_xi = K_gain_pose @ innovation
            delta_T = twist2pose(axangle2twist(delta_xi.reshape(1, 6)))[0]
            T_curr = T_curr @ delta_T 
            curr_T_inv = np.linalg.inv(T_curr)

            # update sigma
            sigma_m_j = (np.eye(3) - K_gain_map @ H_map_j) @ sigma_m_j
            sigma_m_j = (sigma_m_j + sigma_m_j.T) / 2.0
            sigma_map[3*j:3*j+3, 3*j:3*j+3] = sigma_m_j 
            sigma_pose = (np.eye(6) - K_gain_pose @ H_pose_j) @ sigma_pose
            sigma_pose = (sigma_pose + sigma_pose.T) / 2.0

        pose_trajectory[t] = T_curr
            
    return pose_trajectory, mu

def visualize_trajectory_landmark(pose, mu=None, path_name="Unknown", show_ori=False):

    fig, ax = plt.subplots(figsize=(5, 5))
    n_pose = pose.shape[0]

    if mu is not None:
        valid_mask = mu[2, :] != 0
        valid_mu = mu[:, valid_mask]
        ax.scatter(valid_mu[0, :], valid_mu[1, :], s=1, c='black', alpha=0.1, label="Landmarks")

    ax.plot(pose[:,0,3], pose[:,1,3], 'r-', label=path_name)
    ax.scatter(pose[0,0,3], pose[0,1,3], marker='s', label="start")
    ax.scatter(pose[-1,0,3], pose[-1,1,3], marker='o', label="end")
  
    if show_ori:
        select_ori_index = list(range(0, n_pose, max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _, _, yaw = mat2euler(pose[i, :3, :3])
            yaw_list.append(yaw)
  
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx, dy = [dx, dy] / np.sqrt(dx**2 + dy**2)
        ax.quiver(pose[select_ori_index, 0, 3], pose[select_ori_index, 1, 3], dx, dy, \
            color="b", units="xy", width=1, headlength=0.002, headaxislength=0.001)
            
    traj_x = pose[:, 0, 3]
    traj_y = pose[:, 1, 3]
    margin = 40
    ax.set_xlim([traj_x.min() - margin, traj_x.max() + margin])
    ax.set_ylim([traj_y.min() - margin, traj_y.max() + margin])
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show()

    return fig, ax

if __name__ == "__main__":
    filepath = "../data/dataset02/dataset02.npy"
    print(f"========== Readfile: {filepath} ==========")
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filepath)
    
    if "02" in filepath:
        features = np.load("../data/dataset02/dataset02_features.npy")

    print("========== Start VI-SLAM ==========")
    final_poses, final_mu = visual_inertial_slam(v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu)
    
    print("========== plot results ==========")
    visualize_trajectory_landmark(final_poses, final_mu, path_name="VI-SLAM Trajectory")