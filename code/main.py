import numpy as np
from pr3_utils import *
from imu_localization import imu_localization
from landmark_mapping import landmark_estimate, visualize_estimate_landmark
from visual_inertial_slam import visual_inertial_slam, visualize_trajectory_landmark


if __name__ == '__main__':

	# Load the measurements
	filename = "../data/dataset02/dataset02.npy"
	v_t,w_t,timestamps,features,K_l,K_r,extL_T_imu,extR_T_imu = load_data(filename)

	# data02 use own feature map
	if "02" in filename:
		print("data02 use own feature map")
		features = np.load("../data/dataset02/dataset02_features.npy")

	# (a) IMU Localization via EKF Prediction

	print("Running (a) IMU Localization via EKF Prediction")
	poses, sigmas = imu_localization(v_t, w_t, timestamps)

	fig, ax = visualize_trajectory_2d(poses, path_name="IMU Localization", show_ori=True)
	plt.show()

	# (b) Landmark Mapping via EKF Update

	print("Running (b) Landmark Mapping via EKF Update")
	estimate_landmark, sigma = landmark_estimate(features, poses, K_l, K_r, extL_T_imu, extR_T_imu)

	fig, ax = visualize_estimate_landmark(pose=poses, mu=estimate_landmark, path_name="IMU Trajectory", show_ori=True)
	plt.show()

	# (c) Visual-Inertial SLAM
	print("Running (c) Visual-Inertial SLAM")
	final_poses, final_mu = visual_inertial_slam(v_t, w_t, timestamps, features, K_l, K_r, extL_T_imu, extR_T_imu)
	visualize_trajectory_landmark(final_poses, final_mu, path_name="VI-SLAM Trajectory")

	# You may use the function below to visualize the robot pose over time
	# visualize_trajectory_2d(world_T_imu, show_ori = True)

	print("========== plot comparison results ==========")
	visualize_comparison(poses, estimate_landmark, final_poses, final_mu, show_ori=False)


