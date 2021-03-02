import numpy as np
import copy


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