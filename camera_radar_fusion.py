import numpy as np
import copy
from nuscenes.utils.data_classes import PointCloud
from pyquaternion import Quaternion


def radar_fusion_min_distance(annotation_data, radar_point):
    selected_point_ids = [None] * len(annotation_data)
    selected_points = [None] * len(annotation_data)
    distances = [None] * len(annotation_data)
    point_dists = np.linalg.norm(radar_point[:2, :], axis=0)
    for idx, data in enumerate(annotation_data):
        radar_indexes = data["radar_indexes"]

        if len(radar_indexes) > 0:
            # fusion
            dists = point_dists[radar_indexes]
            distances[idx] = dists.min()
            selected_point_id = dists.argmin()
            selected_point_ids[idx] = radar_indexes[selected_point_id]
            selected_points[idx] = radar_point[:2,radar_indexes[selected_point_id]]

    return selected_point_ids, selected_points, distances


def radar_fusion_cluster(annotation_data, radar_point):
    num_bbox = len(annotation_data)
    selected_point_ids = [None] * num_bbox
    selected_points = [None] * num_bbox
    distances = [None] * num_bbox
    radar_points_in_bboxes = list(set(sum([data["radar_indexes"] for data in annotation_data], []))) # bboxに入った点を抽出
    point_xyz_in_bboxes = radar_point[:,radar_points_in_bboxes].copy().transpose()
    point_dists_sq = np.square(radar_point[:2,radar_points_in_bboxes]).sum(axis=0) # radarと各点の距離を計算
    near_radius = 3.0 # 同一オブジェクトとみなす半径
    near_radius_sq = near_radius**2
    assigned_bbox = []
    
    annotation_data_cp = copy.deepcopy(annotation_data)

    if len(point_dists_sq) > 0:
        idxs_and_points = zip(point_dists_sq, radar_points_in_bboxes, point_xyz_in_bboxes)
        idxs_and_points = sorted(idxs_and_points)
        point_dists_sq, radar_points_in_bboxes, point_xyz_in_bboxes = zip(*idxs_and_points) # radarに近い順にソート
        point_xyz_in_bboxes = np.array(point_xyz_in_bboxes)

        point_dists_sq = list(point_dists_sq)
        radar_points_in_bboxes = list(radar_points_in_bboxes)
        num_radar_points = len(radar_points_in_bboxes)

        while num_radar_points != 0:
            pivot_point = 0 # radarに近い点から探してく
            for bbox_idx_, data in enumerate(annotation_data_cp):
                if not (bbox_idx_ in assigned_bbox):
                    if radar_points_in_bboxes[pivot_point] in data["radar_indexes"]:
                        distances[bbox_idx_] = np.sqrt(point_dists_sq[pivot_point]) # オブジェクトの距離
                        selected_point_ids[bbox_idx_] = radar_points_in_bboxes[pivot_point] # 採用点のid
                        selected_points[bbox_idx_] = point_xyz_in_bboxes[pivot_point, :2] # 採用点の位置
                        assigned_bbox.append(bbox_idx_)
                        break

            # 自身のrange プラマイ near_radius [m]に絞る
            range_near_ponit_indexes = list(np.where(point_dists_sq[pivot_point] - near_radius_sq < point_dists_sq)[0]) + list(np.where(point_dists_sq < point_dists_sq[pivot_point] + near_radius_sq)[0])
            range_near_ponit_indexes = sorted(set(range_near_ponit_indexes), key = range_near_ponit_indexes.index) # 重複を削除
            near_points_xy = point_xyz_in_bboxes[range_near_ponit_indexes, :2] - point_xyz_in_bboxes[pivot_point, :2].reshape((1,2)) # 自身から各点までの位置ベクトル
            near_points = list(np.where(np.linalg.norm(near_points_xy, axis=1) < near_radius)[0]) # 自身から球形near_radius以内にいる点のインデックス
            for rm_cnt, near_point in enumerate(near_points):
                radar_points_in_bboxes.pop(near_point - rm_cnt) # near_radius以内の点を削除
                point_dists_sq.pop(near_point - rm_cnt) # point_dists_sqのほうも消す
                point_xyz_in_bboxes = np.delete(point_xyz_in_bboxes, near_point - rm_cnt, axis=0)

            num_radar_points = len(radar_points_in_bboxes)
    
    return selected_point_ids, selected_points, distances


def radar_fusion_camera_to_bev(annotation_data, radar_point, nusc, token, heights):
    num_bbox = len(annotation_data)
    selected_point_ids = [None] * num_bbox
    selected_points = [None] * num_bbox
    distances = [None] * num_bbox

    sample = nusc.get('sample', token)
    camera_rec = nusc.get('sample_data', sample['data']["CAM_FRONT"])
    radar_rec = nusc.get('sample_data', sample['data']["RADAR_FRONT"])
    cs_record = nusc.get('calibrated_sensor', camera_rec['calibrated_sensor_token'])
    camera_intrinsics = np.array(cs_record['camera_intrinsic'])
    inv_intrinsics = np.linalg.inv(camera_intrinsics)

    if len(annotation_data) == 0:
        return selected_point_ids, selected_points, distances

    centers = []
    for data in annotation_data:
        bbox = data["box"]
        label = data["label"]
        object_height = heights[label] # クラス毎に高さを変える

        left_lower = np.array([bbox[0], bbox[3], 1]) # u, v, 1
        left_upper = np.array([bbox[0], bbox[1], 1]) # u, v, 1
        right_lower = np.array([bbox[2], bbox[3], 1]) # u, v, 1
        left_lower_xyz = np.dot(inv_intrinsics, left_lower)
        left_upper_xyz = np.dot(inv_intrinsics, left_upper)
        right_lower_xyz = np.dot(inv_intrinsics, right_lower)
        xyz_height = left_upper_xyz[1] - left_lower_xyz[1]
        scale = np.abs(object_height / xyz_height)
        low_center = scale * np.array([0.5*(right_lower_xyz[0] + left_lower_xyz[0]), left_lower_xyz[1], 1])[:,np.newaxis]
        centers.append(low_center)
    centers = np.concatenate(centers, axis=1)
    pc = PointCloud(centers)
    
    # camera to world
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # world to ego vehicle
    poserecord = nusc.get('ego_pose', camera_rec['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # egho vehicle to world
    poserecord = nusc.get('ego_pose', radar_rec['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # world to radar
    cs_record = nusc.get('calibrated_sensor', radar_rec['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    bev_points = pc.points[:2,:]
    for idx, data in enumerate(annotation_data):
        bev_point = bev_points[:, idx]
        p2p_distances = np.square(radar_point[:2,:] - bev_point[:,np.newaxis]).sum(axis=0)
        min_distance_idx = p2p_distances.argmin()
        selected_point_ids[idx] = min_distance_idx
        #selected_points[idx] = radar_point[:2,idx]
        selected_points[idx] = bev_point
        #distances[idx] = np.linalg.norm(radar_point[:2,idx], axis=0)
        distances[idx] = np.linalg.norm(bev_point, axis=0)
    
    return selected_point_ids, selected_points, distances


def fusion_bevxyz_and_radar_min(annotation_data, radar_point, nusc, token):
    num_bbox = len(annotation_data)
    selected_point_ids = [None] * num_bbox
    selected_points = [None] * num_bbox
    distances = [None] * num_bbox

    radar_selected_ids, radar_selected_points, radar_distances = radar_fusion_min_distance(annotation_data, radar_point)
    camera_selected_ids, camera_selected_points, camera_distances = radar_fusion_camera_to_bev(annotation_data, radar_point, nusc, token)

    distances = [camera_distances[idx] if radar_distance is None else min(camera_distances[idx], radar_distances[idx]) for idx, radar_distance in enumerate(radar_distances)]
    selected_points = [camera_selected_points[idx] if radar_distance is None else (camera_selected_points[idx] if np.argmin([camera_distances[idx], radar_distances[idx]]) == 0 else radar_selected_points[idx]) for idx, radar_distance in enumerate(radar_distances)]

    selected_point_ids = radar_selected_ids

    return selected_point_ids, selected_points, distances