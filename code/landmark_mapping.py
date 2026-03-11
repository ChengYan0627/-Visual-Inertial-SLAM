import numpy as np
from pr3_utils import load_data
import scipy.sparse as sparse
from imu_localization import imu_localization 
import matplotlib.pyplot as plt
from pr3_utils import mat2euler 

def initialize_landmark(z, T, Ks, leftcam_T_imu):
    f_u, c_u, f_v, c_v = Ks[0,0], Ks[0,2], Ks[1,1], Ks[1,2]
    fu_b = -Ks[2, 3] 
    disparity = z[0] - z[2]
    if disparity <= 1.0:
        return None
    Z = fu_b / disparity
    X = (z[0] - c_u) * Z / f_u
    Y = (z[1] - c_v) * Z / f_v
    P_cam = np.array([X, Y, Z, 1])
    m_world = T @ np.linalg.inv(leftcam_T_imu) @ P_cam
    return m_world[:3]

def jacobian_H(q, K, cam_T_imu, T_inv, P_T):

    x, y, z, _ = q
    dpi_dq = (1.0 / z) * np.array([
        [1, 0, -x/z, 0],
        [0, 1, -y/z, 0],
        [0, 0,    0, 0]  
    ])
    
    H_full = K @ dpi_dq @ cam_T_imu @ T_inv @ P_T
    return H_full[:2, :]

def landmark_estimate(z_t, all_T, K_l, K_r, extL_T_imu, extR_T_imu):
    _, features_M, total_t = z_t.shape
    
    mu = np.zeros((3, features_M))
    sigma = sparse.eye(3 * features_M, format='csr') * 0.1
        
    initialized = np.zeros(features_M, dtype=bool)
    R = np.eye(4) * 5.0                  
    P_T = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]]) 

    # Ks
    baseline = np.linalg.norm(extL_T_imu[:3, 3] - extR_T_imu[:3, 3])
    Ks_approx = np.array([
        [K_l[0,0], 0, K_l[0,2], 0],
        [0, K_l[1,1], K_l[1,2], 0],
        [K_l[0,0], 0, K_l[0,2], -K_l[0,0] * baseline],
        [0, K_l[1,1], K_l[1,2], 0]
    ])

    I_sparse = sparse.eye(3 * features_M, format='csr')

    for t in range(total_t):
        curr_T = all_T[t]
        curr_T_inv = np.linalg.inv(curr_T) 
        
        for j in range(features_M):
            z = z_t[:, j, t]
            if z[0] == -1: 
                continue 
            
            # first time see feature
            if not initialized[j]:
                init_m = initialize_landmark(z, curr_T, Ks_approx, extL_T_imu)

                # too far
                if init_m is None: 
                    continue 

                mu[:, j] = init_m
                initialized[j] = True
                continue
                
            mu_homo = np.append(mu[:, j], 1)
            

            # world to camera frame
            q_L = extL_T_imu @ curr_T_inv @ mu_homo
            q_R = extR_T_imu @ curr_T_inv @ mu_homo
        
            if q_L[2] <= 0 or q_R[2] <= 0:
                continue
            
            # camera frame to pixel 
            u_L = K_l[0,0] * (q_L[0] / q_L[2]) + K_l[0,2]
            v_L = K_l[1,1] * (q_L[1] / q_L[2]) + K_l[1,2]
            
            u_R = K_r[0,0] * (q_R[0] / q_R[2]) + K_r[0,2]
            v_R = K_r[1,1] * (q_R[1] / q_R[2]) + K_r[1,2]
            
            # estimated pixel 
            z_hat = np.array([u_L, v_L, u_R, v_R])
            
            # Jacobian
            H_L = jacobian_H(q_L, K_l, extL_T_imu, curr_T_inv, P_T)
            H_R = jacobian_H(q_R, K_r, extR_T_imu, curr_T_inv, P_T)
            H_j = np.vstack((H_L, H_R)) # 4x3 
            
            # # ==========================================
            # # EKF Update
            # H_sparse = sparse.lil_matrix((4, 3 * features_M))
            # H_sparse[:, 3*j : 3*j+3] = H_j
            # H_sparse = H_sparse.tocsr()
            
            # S = (H_sparse @ sigma @ H_sparse.T).toarray() + R
            # K_gain = sigma @ H_sparse.T @ np.linalg.inv(S)
            
            # innovation = z - z_hat
            # update_vector = (K_gain @ innovation)
            # mu[:, j] += update_vector[3*j : 3*j+3]
            
            # sigma = (I_sparse - K_gain @ H_sparse) @ sigma

            # ==========================================
            # EKF Update
            H_sparse = sparse.lil_matrix((4, 3 * features_M))
            H_sparse[:, 3*j : 3*j+3] = H_j
            H_sparse = H_sparse.tocsr()
            
            S_sparse = H_sparse @ sigma @ H_sparse.T
            if sparse.issparse(S_sparse):
                S = S_sparse.toarray() + R
            else:
                S = np.asarray(S_sparse) + R
            
            S_inv_sparse = sparse.csr_matrix(np.linalg.inv(S))
            
            K_gain = sigma @ H_sparse.T @ S_inv_sparse
            
            innovation = z - z_hat
            
            update_vector = K_gain.dot(innovation)
            update_vector = np.asarray(update_vector).flatten()
            
            mu[:, j] += update_vector[3*j : 3*j+3]
            
            sigma = (I_sparse - K_gain @ H_sparse) @ sigma
            
    return mu, sigma

def visualize_estimate_landmark(pose, mu=None, path_name="Unknown", show_ori=False):

    fig, ax = plt.subplots(figsize=(5, 5))
    n_pose = pose.shape[0]

    if mu is not None:
        valid_mask = mu[2, :] != 0
        valid_mu = mu[:, valid_mask]
        ax.scatter(valid_mu[0, :], valid_mu[1, :], s=1, c='black', alpha=0.1, label="Landmarks")
        # ax.scatter(valid_mu[0, :], valid_mu[1, :], s=1, c='green', alpha=0.1, label="Landmarks")

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

    return fig, ax

def run_landmark_ekf(filepath, features, poses, K_l, K_r, extL_T_imu, extR_T_imu):

    if "02" in filepath:
        print("data02 use own feature map")
        features = np.load("../data/dataset02/dataset02_features.npy")

    print("Start EKF")
    mu, sigma = landmark_estimate(features, poses, K_l, K_r, extL_T_imu, extR_T_imu)

    return mu, sigma

if __name__ == "__main__":
        
    filepath = "../data/dataset02/dataset02.npy"
    v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu = load_data(filepath)
    print(f"========== running {filepath} ==========")
    
    # data02 use own feature map
    if "02" in filepath:
        print("data02 use own feature map")
        features = np.load("../data/dataset02/dataset02_features.npy")

    poses, sigmas = imu_localization(v_t, w_t, timestamps)
    mu, sigma = landmark_estimate(features, poses, K_l, K_r, extL_T_imu, extR_T_imu)
    print(f"mu shape: {mu.shape}")
    print(f"sigma shape: {sigma.shape}, type: {type(sigma)}")
    fig, ax = visualize_estimate_landmark(pose=poses, mu=mu, path_name="IMU Trajectory", show_ori=True)
    plt.show()