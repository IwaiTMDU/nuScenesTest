from nuscenes.nuscenes import NuScenes
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
import cv2
import io


dataroot = "v1.0-mini"
class_to_color = {
        'bg': np.array([0, 0, 0])/255,
        'human.pedestrian.adult': np.array([34, 114, 227]) / 255,
        'vehicle.bicycle': np.array([0, 182, 0])/255,
        'vehicle.bus': np.array([84, 1, 71])/255,
        'vehicle.car': np.array([189, 101, 0]) / 255,
        'vehicle.motorcycle': np.array([159, 157,156])/255,
        'vehicle.trailer': np.array([0, 173, 162])/255,
        'vehicle.truck': np.array([89, 51, 0])/255,
        }

def put_bbox_into_image(annotation):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.4

    image = cv2.imread(annotation["image_file"])
    for data in annotation["annotations"]:
        if not (data["label"] in class_to_color):
            continue
        (x1, y1, x2, y2) = data["box"].astype(np.int32)
        color =class_to_color[data["label"]] * 255
        cv2.rectangle(image,(x1, y1), (x2, y2), color,2)

    return image

def get_annotation_bbox(nusc, tokens):
    from nuscenes.utils.geometry_utils import BoxVisibility
    annotations = []
    for token_index in tqdm(range(len(tokens))):
        token = tokens[token_index]
        sample = nusc.get('sample', token)
        camera_rec = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        radar_rec = nusc.get('sample_data', sample['data']["RADAR_FRONT"])
        camera_pos = nusc.get('calibrated_sensor', camera_rec['calibrated_sensor_token'])['translation']
        radar_pos = nusc.get('calibrated_sensor', radar_rec['calibrated_sensor_token'])['translation']
        camera_to_radar_pos = np.array(camera_pos) - np.array(radar_pos)
        _, boxes, camera_intrinsic = nusc.get_sample_data(camera_rec['token'], box_vis_level=BoxVisibility.ANY)

        bboxes = []
        for box in boxes:
            corners = box.bottom_corners()
            corners = corners[[2,0,1], :]
            corners[1,:] = -corners[1,:]
            bev_corners = corners + np.expand_dims(camera_to_radar_pos, axis=1)
            camera_bbox = box.box2d(camera_intrinsic)
            
            bboxes.append({"label":box.name, "box":camera_bbox, "bev_box":bev_corners})
        annotations.append({"image_file":os.path.join(dataroot,camera_rec["filename"]), "annotations":bboxes})

    return annotations


def get_radar_points(nusc, tokens):
    radar_points = []
    for token_index in tqdm(range(len(tokens))):
        token = tokens[token_index]
        sample = nusc.get('sample', token)
        radar_data = nusc.get('sample_data', sample['data']["RADAR_FRONT"])
        pcd = o3d.io.read_point_cloud(os.path.join(dataroot, radar_data['filename']))
        np_pcd = np.array(pcd.points)
        radar_points.append(np_pcd)

    return radar_points

if __name__ == "__main__":
    nusc = NuScenes(version = "v1.0-mini", dataroot = dataroot, verbose=False)
    scene_num = len(nusc.scene)
    sample_tokens = {}
    prog = 0
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)
    
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
    radar_points = get_radar_points(nusc, sample_tokens)

    # BEV
    plt.figure(1)
    scatter_size = 5
    for token_index in tqdm(range(len(sample_tokens))):
        radar_point = radar_points[token_index]

        plt.clf()
        plt.xlabel("y [m]")
        plt.ylabel("x(forward) [m]")
        plt.scatter(-radar_point[:,1], radar_point[:,0], color = "black", s = scatter_size)
        
        for data in annotations[token_index]["annotations"]:
            if not (data["label"] in class_to_color):
                continue
            color = class_to_color[data["label"]]
            corner = data["bev_box"]
            for i_corner in range(4):
                plt.plot([-corner[1][i_corner], -corner[1][(i_corner+1)%4]], [corner[0][i_corner], corner[0][(i_corner+1)%4]], 'k-', c = np.concatenate([color, [1]]))  
        
        bev_im_buf = io.BytesIO()
        plt.savefig(bev_im_buf, format='png', bbox_inches='tight')
        bev_im = cv2.imdecode(np.frombuffer(bev_im_buf.getvalue(), dtype=np.uint8), 1)
        bev_im = bev_im[:,:,::-1]
        cam_img = put_bbox_into_image(annotations[token_index])
        cam_img = cv2.resize(cam_img, (640, 320))
        asp = bev_im.shape[0] / cam_img.shape[0]
        bev_im = cv2.resize(bev_im, dsize=(round(asp*bev_im.shape[1]), cam_img.shape[0]))
        out_img = cv2.hconcat([cam_img, bev_im])
        cv2.imwrite(os.path.join(save_dir, str(token_index).zfill(4)+".png"), out_img)