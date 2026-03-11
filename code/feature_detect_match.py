import numpy as np
import cv2

def feature_detect_match(left_frames, right_frames):

    T = len(left_frames)
    
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    feature_params = dict(maxCorners=100, qualityLevel=0.01, minDistance=15, blockSize=7)
    
    features_dict = []       # [{id: (lx, ly, rx, ry)}, ...]
    active_left_feat = None 
    active_left_ids = []        
    new_feature_id = 0   
    
    # =========================================================
    # Time t=0
    curr_left_img = np.array(left_frames[0], dtype=np.uint8)
    curr_right_img = np.array(right_frames[0], dtype=np.uint8)

    # ==== feature video ====
    height, width = curr_left_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_left = cv2.VideoWriter('../data/dataset02/tracking_result_left.mp4', fourcc, 20.0, (width, height))
    video_right = cv2.VideoWriter('../data/dataset02/tracking_result_right.mp4', fourcc, 20.0, (width, height))
    # ==================
    
    # find corners (goodFeaturesToTrack)
    p0_left = cv2.goodFeaturesToTrack(curr_left_img, mask=None, **feature_params)
    
    # find left corners in right (calcOpticalFlowPyrLK)
    p0_right, status_stereo, _ = cv2.calcOpticalFlowPyrLK(curr_left_img, curr_right_img, p0_left, None, **lk_params)
    
    # delete fail points status_stereo == 0
    good_mask = (status_stereo.flatten() == 1) & (np.abs(p0_left[:, 0, 1] - p0_right[:, 0, 1]) < 2.0)
    
    active_left_feat = p0_left[good_mask]
    active_right_feat = p0_right[good_mask]
    
    # give ids
    active_left_ids = np.arange(new_feature_id, new_feature_id + len(active_left_feat))
    new_feature_id += len(active_left_feat)
    
    # build z_0
    z_0_dict = {}
    for i, features_id in enumerate(active_left_ids):
        lx, ly = active_left_feat[i, 0]
        rx, ry = active_right_feat[i, 0]
        z_0_dict[features_id] = (lx, ly, rx, ry)
    features_dict.append(z_0_dict)

    prev_left = curr_left_img
    
    # Time t=1, 2, 3...
    for t in range(1, T):
        curr_left_img = np.array(left_frames[t], dtype=np.uint8)
        curr_right_img = np.array(right_frames[t], dtype=np.uint8)
        z_t_dict = {}
        
        # find Left t-1 corners in Left t (calcOpticalFlowPyrLK)
        p1_left, status_time, _ = cv2.calcOpticalFlowPyrLK(prev_left, curr_left_img, active_left_feat, None, **lk_params)
        time_mask = (status_time.flatten() == 1)
        
        # remove points not in frame
        p1_left_survived = p1_left[time_mask]
        ids_survived = active_left_ids[time_mask]
        
        # # find left corners in right (calcOpticalFlowPyrLK)
        p1_right, status_stereo, _ = cv2.calcOpticalFlowPyrLK(curr_left_img, curr_right_img, p1_left_survived, None, **lk_params)
        
        # delete fail points status_stereo == 0
        stereo_mask = (status_stereo.flatten() == 1) & (np.abs(p1_left_survived[:, 0, 1] - p1_right[:, 0, 1]) < 2.0)
        
        active_left_feat = p1_left_survived[stereo_mask]
        active_right_feat = p1_right[stereo_mask]
        active_left_ids = ids_survived[stereo_mask]
        
        # z_t
        for i, features_id in enumerate(active_left_ids):
            lx, ly = active_left_feat[i, 0]
            rx, ry = active_right_feat[i, 0]
            z_t_dict[features_id] = (lx, ly, rx, ry)

        features_dict.append(z_t_dict)

        # =========================================================
        # Visualization

        # left
        vis_frame = cv2.cvtColor(curr_left_img, cv2.COLOR_GRAY2BGR)
        
        if active_left_feat is not None and active_right_feat is not None:
            # same ID
            for pt_l, pt_r in zip(active_left_feat, active_right_feat):
                lx, ly = int(pt_l[0, 0]), int(pt_l[0, 1])
                rx, ry = int(pt_r[0, 0]), int(pt_r[0, 1])
                
                # dis
                cv2.line(vis_frame, (lx, ly), (rx, ry), (0, 255, 0), 1)
                
                # right feature (blue)
                cv2.circle(vis_frame, (rx, ry), 3, (255, 0, 0), -1)
                
                # left corner (red)
                cv2.circle(vis_frame, (lx, ly), 3, (0, 0, 255), -1)
                
        video_left.write(vis_frame)
        
        if t % 100 == 0:
            print(f"processing left: {t} / {T} frame")

        # right
        vis_frame = cv2.cvtColor(curr_right_img, cv2.COLOR_GRAY2BGR)
        
        if active_left_feat is not None and active_right_feat is not None:
            # same ID
            for pt_l, pt_r in zip(active_left_feat, active_right_feat):
                lx, ly = int(pt_l[0, 0]), int(pt_l[0, 1])
                rx, ry = int(pt_r[0, 0]), int(pt_r[0, 1])
                
                # dis
                cv2.line(vis_frame, (lx, ly), (rx, ry), (0, 255, 0), 1)
                
                # right feature (blue)
                cv2.circle(vis_frame, (rx, ry), 3, (255, 0, 0), -1)
                
                # left corner (red)
                cv2.circle(vis_frame, (lx, ly), 3, (0, 0, 255), -1)
                
        video_right.write(vis_frame)
        
        if t % 100 == 0:
            print(f"processing right: {t} / {T} frame")
        
        # =========================================================
        # introduce more feature
        if len(active_left_feat) < 50:
            mask = np.ones_like(curr_left_img, dtype=np.uint8) * 255
            for pt in active_left_feat:
                cv2.circle(mask, (int(pt[0,0]), int(pt[0,1])), 15, 0, -1) 
            
            new_left_feat = cv2.goodFeaturesToTrack(curr_left_img, mask=mask, **feature_params)
            
            if new_left_feat is not None:

                new_ids = np.arange(new_feature_id, new_feature_id + len(new_left_feat))
                new_feature_id += len(new_left_feat)
                
                active_left_feat = np.vstack((active_left_feat, new_left_feat))
                active_left_ids = np.concatenate((active_left_ids, new_ids))
        
        prev_left = curr_left_img
    
    video_right.release()
    video_left.release()
    print("saved videos")

    # =========================================================
    # 4 x M x T 

    M = new_feature_id
    z = np.full((4, M, T), -1.0)
    
    for t in range(T):
        for features_id, (lx, ly, rx, ry) in features_dict[t].items():
            z[0, features_id, t] = lx
            z[1, features_id, t] = ly
            z[2, features_id, t] = rx
            z[3, features_id, t] = ry
            
    return z