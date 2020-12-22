from nuscenes.nuscenes import NuScenes
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_annotation_bbox(nusc, tokens):
    from nuscenes.utils.geometry_utils import BoxVisibility
    bboxes = []
    for token_index in tqdm(range(len(tokens))):
        token = tokens[token_index]
        sample = nusc.get('sample', token)
        camera_rec = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        radar_rec = nusc.get('sample_data', sample['data']["RADAR_FRONT"])
        camera_pos = nusc.get('calibrated_sensor', camera_rec['calibrated_sensor_token'])['translation']
        radar_pos = nusc.get('calibrated_sensor', radar_rec['calibrated_sensor_token'])['translation']
        camera_to_radar_pos = np.array(camera_pos) - np.array(radar_pos)
        _, boxes, camera_intrinsic = nusc.get_sample_data(camera_rec['token'], box_vis_level=BoxVisibility.ANY)

        for box in boxes:
            corners = box.bottom_corners()
            corners = corners[[2,0,1], :]
            corners[1,:] = -corners[1,:]
            bev_corners = corners + np.expand_dims(camera_to_radar_pos, axis=1)
            camera_bbox = box.box2d(camera_intrinsic)
            
            bboxes.append({"label":box.name, "box":camera_bbox, "bev_box":bev_corners})

    return bboxes


if __name__ == "__main__":
    dataroot = "v1.0-mini"
    nusc = NuScenes(version = "v1.0-mini", dataroot = dataroot, verbose=False)
    scene_num = len(nusc.scene)
    sample_tokens = {}
    prog = 0
    
    # Get all tokens
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

    annotations = get_annotation_bbox(nusc, sample_tokens)
    print(annotations)