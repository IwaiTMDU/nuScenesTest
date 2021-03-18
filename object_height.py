from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, PointCloud
from nuscenes.utils.geometry_utils import BoxVisibility
import numpy as np
import os
import tqdm
from pyquaternion import Quaternion
#import torch
import mxnet as mx 
import random


version = "v1.0-mini"
dataroot = "v1.0-mini"
box_visibility = BoxVisibility.ANY
class_names = [
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck'
]


def get_bboxes(nusc, tokens):
    token_bboxes = []
    for token_index in tqdm.tqdm(range(len(tokens))):
        bboxes = []
        token = tokens[token_index]
        sample = nusc.get('sample', token)
        camera_rec = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        _, boxes, camera_intrinsic = nusc.get_sample_data(camera_rec['token'], box_vis_level=box_visibility)

        for box in boxes:
            if box.name in class_names:
                corners = box.bottom_corners()
                #corners = corners[[2,0,1], :]
                
                bev_center = np.array(corners).mean(axis=1)[[0,2]]
                camera_bbox = box.box2d(camera_intrinsic)
                bboxes.append({"label":box.name, "box":camera_bbox, "bev_center":bev_center})
        token_bboxes.append(bboxes)
    return token_bboxes


def calc_lowcenter_and_height(token_bboxes, objects_height, nusc, tokens):
    low_center_norms = [None] * len(objects_height)
    xyz_heights = [None] * len(objects_height)
    gt_bev_centers = [None] * len(objects_height)
    object_idxs = []
    
    for token_index in range(len(tokens)):
        token = tokens[token_index]
        sample = nusc.get('sample', token)
        camera_rec = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        cs_record = nusc.get('calibrated_sensor', camera_rec['calibrated_sensor_token'])
        camera_intrinsics = np.array(cs_record['camera_intrinsic'])
        inv_intrinsics = np.linalg.inv(camera_intrinsics)

        if len(token_bboxes[token_index]) == 0:
            continue

        for bbox in token_bboxes[token_index]:
            _bbox = bbox["box"]
            label = bbox["label"]
            object_idx = class_names.index(label)
            object_idxs.append(object_idx)

            left_lower = np.array([_bbox[0], _bbox[3], 1]) # u, v, 1
            left_upper = np.array([_bbox[0], _bbox[1], 1]) # u, v, 1
            right_lower = np.array([_bbox[2], _bbox[3], 1]) # u, v, 1
            left_lower_xyz = np.dot(inv_intrinsics, left_lower)
            left_upper_xyz = np.dot(inv_intrinsics, left_upper)
            right_lower_xyz = np.dot(inv_intrinsics, right_lower)
            low_center_norm = np.array([0.5*(right_lower_xyz[0] + left_lower_xyz[0]), 1])[np.newaxis,:]
            xyz_height = left_lower_xyz[1] - left_upper_xyz[1]

            if low_center_norms[object_idx] is None:
                low_center_norms[object_idx] = low_center_norm
            else:
                low_center_norms[object_idx] = np.concatenate([low_center_norms[object_idx], low_center_norm], axis=0)

            if xyz_heights[object_idx] is None:
                xyz_heights[object_idx] = np.array([xyz_height])[np.newaxis, :]
            else:
                xyz_heights[object_idx] = np.concatenate([xyz_heights[object_idx], np.array([xyz_height])[np.newaxis, :]], axis=0)

            if gt_bev_centers[object_idx] is None:
                gt_bev_centers[object_idx] = bbox["bev_center"][np.newaxis, :]
            else:
                gt_bev_centers[object_idx] = np.concatenate([gt_bev_centers[object_idx], bbox["bev_center"][np.newaxis, :]], axis=0)

    return low_center_norms, np.array(xyz_heights), gt_bev_centers, object_idxs


def calc_object_distance_error(objects_height, low_center_norms, xyz_heights, object_idxs, gt_bev_centers):
    obj_num = len(class_names)
    min_object_num = 1000000000
    for obj_id in range(obj_num):
        min_object_num = min(min_object_num, low_center_norms[obj_id].shape[0])
    batch_size = min(3, min_object_num)

    optim_objects_height = [0]*obj_num
    min_obj_distance_error = [0]*obj_num
    valid = [False] * obj_num

    for num in tqdm.tqdm(range(500)):
        sample_idxs = [random.sample(range(low_center_norms[obj_id].shape[0]), batch_size) for obj_id in range(obj_num)]
        
        heights_tensor = mx.nd.array(objects_height)
        low_center_tensor = mx.nd.array([low_center_norms[obj_id][sample_idxs[obj_id]] for obj_id in range(obj_num)])
        xyz_heights_tensor = mx.nd.array([xyz_heights[obj_id][sample_idxs[obj_id]] for obj_id in range(obj_num)]).squeeze(axis=2)
        gt_center_tensor = mx.nd.array([gt_bev_centers[obj_id][sample_idxs[obj_id]] for obj_id in range(obj_num)])

        heights_tensor.attach_grad()
        optim = mx.optimizer.SGD(learning_rate = 0.0001)
        op_state = optim.create_state(0, heights_tensor)
        for i in range(3000):
            with mx.autograd.record():
                scales = mx.nd.broadcast_div(heights_tensor.expand_dims(axis=1), xyz_heights_tensor).abs()
                bev_center = mx.nd.broadcast_mul(scales.expand_dims(axis=2), low_center_tensor)
                distance_error = (gt_center_tensor - bev_center).square().sum(axis=2).mean(axis=1)
                distance_error.backward()
                if distance_error.mean().asscalar() < 1e-6:
                    break
            optim.update(0, heights_tensor, heights_tensor.grad, op_state)

        objects_heights_res = heights_tensor.asnumpy().copy()
        for obj_id in range(obj_num):
            scale = np.abs(objects_heights_res[obj_id] / xyz_heights[obj_id])
            bev_center = scale * low_center_norms[obj_id]
            distance_error = np.linalg.norm(gt_bev_centers[obj_id] - bev_center,axis=1).mean()

            if np.isnan(distance_error):
                continue

            if not valid[obj_id]:
                optim_objects_height[obj_id] = abs(objects_heights_res[obj_id])
                min_obj_distance_error[obj_id] = distance_error
                valid[obj_id] = True
            elif min_obj_distance_error[obj_id] > distance_error:
                optim_objects_height[obj_id] = abs(objects_heights_res[obj_id])
                min_obj_distance_error[obj_id] = distance_error
    
        print("heights : {}".format(optim_objects_height))
        print("mean distance error : {}".format(min_obj_distance_error))
    print("heights : {}".format(optim_objects_height))
    print("mean distance error : {}".format(min_obj_distance_error))

def calc_distance_error(token_bboxes, objects_height, nusc, tokens):
    gt_bev_points = []
    bev_centers = None
    
    for token_index in range(len(tokens)):
        token = tokens[token_index]
        sample = nusc.get('sample', token)
        camera_rec = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        radar_rec = nusc.get('sample_data', sample['data']["RADAR_FRONT"])
        cs_record = nusc.get('calibrated_sensor', camera_rec['calibrated_sensor_token'])
        camera_intrinsics = np.array(cs_record['camera_intrinsic'])
        inv_intrinsics = np.linalg.inv(camera_intrinsics)
        centers = []

        if len(token_bboxes[token_index]) == 0:
            continue

        for bbox in token_bboxes[token_index]:
            gt_bev_points.append(bbox["bev_box"])
            _bbox = bbox["box"]
            label = bbox["label"]
            object_height = objects_height[label]

            left_lower = np.array([_bbox[0], _bbox[3], 1]) # u, v, 1
            left_upper = np.array([_bbox[0], _bbox[1], 1]) # u, v, 1
            right_lower = np.array([_bbox[2], _bbox[3], 1]) # u, v, 1
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

        if bev_centers is None:
            bev_centers = pc.points[:2,:]
        else:
            bev_centers = np.concatenate([bev_centers, pc.points[:2,:]], axis=1)

    bev_centers = bev_centers.transpose()
    gt_bev_points = np.array(gt_bev_points)[:,:2,:]
    gt_bev_centers = gt_bev_points.mean(axis=2)
    distances = np.sum(np.linalg.norm(gt_bev_centers - bev_centers, axis=1))
    print(distances)


if __name__ == '__main__':
    nusc = NuScenes(version = version, dataroot = dataroot, verbose=False)
    scene_num = len(nusc.scene)
    sample_tokens = {}

    # Get all tokens
    prog = 0
    for scene_index in range(scene_num):
        first_sample_token = nusc.scene[scene_index]['first_sample_token']
        nbr_samples = nusc.scene[scene_index]['nbr_samples']
        curr_sample = nusc.get('sample', first_sample_token)

        for _ in range(nbr_samples):
            sample_tokens[prog] = curr_sample['token']
            if curr_sample['next']:
                next_token = curr_sample['next']
                curr_sample = nusc.get('sample', next_token)
            prog += 1
    token_bboxes = get_bboxes(nusc, sample_tokens)

    # Initialize height    
    objects_height = []
    for idx in range(len(class_names)):
        objects_height.append(1.5)
    objects_height = np.array(objects_height)
    
    low_center_norms, xyz_heights, gt_bev_centers, object_idxs = calc_lowcenter_and_height(token_bboxes = token_bboxes, objects_height = objects_height, nusc = nusc, tokens = sample_tokens)
    calc_object_distance_error(objects_height, low_center_norms, xyz_heights, object_idxs, gt_bev_centers)