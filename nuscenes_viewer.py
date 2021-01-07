from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, PointCloud
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import open3d as o3d
import cv2
import io
from PIL import Image
from pyquaternion import Quaternion
import copy


dataroot = "v1.0-mini"
'''
class_names = [
    'bg' ,
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack *',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck'
]
'''

class_names = [
    'bg' ,
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck'
]

class_to_color = {}

def put_bbox_into_image(annotation, radar_in_image = None, rcs_colors = None):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1.0

    image = cv2.imread(annotation["image_file"])
    
    for data in annotation["annotations"]:
        if not (data["label"] in class_to_color):
            continue
        (x1, y1, x2, y2) = data["box"].astype(np.int32)
        color =class_to_color[data["label"]] * 255
        cv2.rectangle(image,(x1, y1), (x2, y2), color,2)
        put_class_label = data["label"].split('.')[-1]
        text = put_class_label
        if data["distance"] is not None:
            text += "[{}m]".format(round(data["distance"], 1))
        (retval,baseLine) = cv2.getTextSize(text, font, fontScale,1)
        textOrg = int(x1), int(y1)

        cv2.rectangle(image, (textOrg[0] - 1,textOrg[1]+baseLine - 1), (textOrg[0]+retval[0] + 1, textOrg[1]-retval[1] - 1), color, -1)
        cv2.putText(image, text, textOrg, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (1,1,1), 1)
        
    if (radar_in_image is not None) and (rcs_colors is not None):
        for point_index in range(radar_in_image.shape[1]):
            point = radar_in_image[:,point_index]
            u, v = int(point[0]), int(point[1])
            point_color = rcs_colors[token_index][point_index]
            point_color = (int(point_color[0]), int(point_color[1]), int(point_color[2]))
            if ((0 <= u <= image.shape[1]) and (0 <= v <= image.shape[0])): # point is in the image
                cv2.circle(image, (u, v), 6, color = point_color, thickness = -1, lineType = cv2.LINE_8)

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
    '''
        RADAR pcd file:
            VERSION 0.7
            FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
            SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
            TYPE F F F I I F F F F F I I I I I I I I
            COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
            WIDTH 125
            HEIGHT 1
            VIEWPOINT 0 0 0 1 0 0 0
            POINTS 125
            DATA binary
            Below some of the fields are explained in more detail:
            x is front, y is left
            vx, vy are the velocities in m/s.
            vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
            We recommend using the compensated velocities.
            invalid_state: state of Cluster validity state.
            (Invalid states)
            0x01	invalid due to low RCS
            0x02	invalid due to near-field artefact
            0x03	invalid far range cluster because not confirmed in near range
            0x05	reserved
            0x06	invalid cluster due to high mirror probability
            0x07	Invalid cluster because outside sensor field of view
            0x0d	reserved
            0x0e	invalid cluster because it is a harmonics
            (Valid states)
            0x00	valid
            0x04	valid cluster with low RCS
            0x08	valid cluster with azimuth correction due to elevation
            0x09	valid cluster with high child probability
            0x0a	valid cluster with high probability of being a 50 deg artefact
            0x0b	valid cluster but no local maximum
            0x0c	valid cluster with high artefact probability
            0x0f	valid cluster with above 95m in near range
            0x10	valid cluster with high multi-target probability
            0x11	valid cluster with suspicious angle
            dynProp: Dynamic property of cluster to indicate if is moving or not.
            0: moving
            1: stationary
            2: oncoming
            3: stationary candidate
            4: unknown
            5: crossing stationary
            6: crossing moving
            7: stopped
            ambig_state: State of Doppler (radial velocity) ambiguity solution.
            0: invalid
            1: ambiguous
            2: staggered ramp
            3: unambiguous
            4: stationary candidates
            pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
            0: invalid
            1: <25%
            2: 50%
            3: 75%
            4: 90%
            5: 99%
            6: 99.9%
            7: <=100%
        returns:
            radar_points
                - Shape: len(tokens) x 3(xyz) x num points
            radar_meta_data
                - Shape: len(tokens) x 15 x num points
                - Semantics:
                    [0]: dyn_prop
                    [1]: id
                    [2]: rcs
                    [3]: vx
                    [4]: vy
                    [5]: vx_comp
                    [6]: vy_comp
                    [7]: is_quality_valid
                    [8]: ambig_state
                    [9]: x_rms
                    [10]: y_rms
                    [11]: invalid_state
                    [12]: pdh0
                    [13]: vx_rms
                    [14]: vy_rms
    '''
    radar_points = []
    radar_meta_data = []
    for token_index in tqdm(range(len(tokens))):
        token = tokens[token_index]
        sample = nusc.get('sample', token)
        radar_sample = nusc.get('sample_data', sample['data']["RADAR_FRONT"])
        radar_data = RadarPointCloud.from_file(os.path.join(dataroot, radar_sample['filename']))
        radar_xyz = radar_data.points[:3,:]
        radar_meta = radar_data.points[3:,:]
        radar_points.append(radar_xyz)
        radar_meta_data.append(radar_meta)

    return radar_points, radar_meta_data

def radar_point_to_image(nusc, tokens, radar_points):
    radar_in_image = []
    for token_index in tqdm(range(len(tokens))):
        token = tokens[token_index]
        sample = nusc.get('sample', token)
        camera_rec = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        radar_rec = nusc.get('sample_data', sample['data']["RADAR_FRONT"])
        pc = PointCloud(copy.deepcopy(radar_points[token_index]))

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', radar_rec['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = nusc.get('ego_pose', radar_rec['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', camera_rec['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = nusc.get('calibrated_sensor', camera_rec['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # [u, v, 1]
        view = np.array(cs_record['camera_intrinsic'])
        points = view_points(pc.points, view, normalize=True).astype(np.int32)
        radar_in_image.append(points)

    return radar_in_image

def get_rcs_color(tokens, radar_meta_data):
    colors = []
    for token_index in tqdm(range(len(tokens))):
        rcs_positive = np.power(10, radar_meta_data[token_index][2,:]/20.0)
        max_rcs = np.max(rcs_positive)

        rcs_hue = 120*(max_rcs - rcs_positive).astype(np.uint8)
        rcs_hsv = 255*np.ones((rcs_hue.shape[0], 3)).astype(np.uint8)
        rcs_hsv[:,0] = rcs_hue
        
        rcs_colors = np.array([cv2.cvtColor(np.array([[rcs_hsv[point_idx]]], dtype=np.uint8), cv2.COLOR_HSV2BGR) for point_idx in range(rcs_hsv.shape[0])])
        rcs_colors = np.reshape(rcs_colors, (rcs_colors.shape[0],3))
        #rcs_colors = np.concatenate([rcs_colors, 255*np.ones((rcs_colors.shape[0],1))], 1)
        colors.append(rcs_colors)

    return colors


def check_radar_in_2dbbox(tokens, annotations, radar_in_image):
    for token_index in tqdm(range(len(tokens))):
        for annotation in annotations[token_index]["annotations"]:
            box = annotation["box"]
            annotation["radar_indexes"] = []
            for point_index in range(radar_in_image[token_index].shape[1]):
                point = radar_in_image[token_index][:,point_index]
                u, v = int(point[0]), int(point[1])
                if((box[0] <= u <= box[2]) and (box[1] <= v <= box[3])):
                    annotation["radar_indexes"].append(point_index)
            
    return annotations


if __name__ == "__main__":
    nusc = NuScenes(version = "v1.0-mini", dataroot = dataroot, verbose=False)
    scene_num = len(nusc.scene)
    sample_tokens = {}
    prog = 0
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)
    distance_errors = []

    # Assign color
    class_to_color['bg'] = np.zeros(3)
    for class_id, class_name in enumerate(class_names):
        class_color_hsv = 255*np.ones(3).astype(np.uint8)
        class_color_hsv[0] = np.uint8(float(class_id) / len(class_names) * 120)
        class_to_color[class_name] = cv2.cvtColor(np.array([[class_color_hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR)/255.0
        class_to_color[class_name] = class_to_color[class_name].reshape(3)
    
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
    radar_points, radar_meta_data = get_radar_points(nusc, sample_tokens)
    radar_in_image = radar_point_to_image(nusc, sample_tokens, radar_points)
    rcs_colors = get_rcs_color(sample_tokens, radar_meta_data);
    annotations = check_radar_in_2dbbox(sample_tokens, annotations, radar_in_image)
    
    out_imgs = []

    # BEV
    plt.figure(1)
    scatter_size = 10
    legends = []
    for label in class_to_color.keys():
        color = class_to_color[label]
        legends.append(mpatches.Patch(color=np.concatenate([color, [1]]), label=label))

    out_vid_shape = (1440, 405)
    out_vid = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 2, out_vid_shape)
    for token_index in tqdm(range(len(sample_tokens))):
        radar_point = radar_points[token_index]

        plt.clf()
        plt.xlabel("y [m]")
        plt.ylabel("x(forward) [m]")
        
        plt.scatter(-radar_point[1,:], radar_point[0,:], c = "black", s = scatter_size)

        vx_vy_comp = radar_meta_data[token_index][5:7,:]*4
        #vx_vy_comp = vx_vy_comp/(np.linalg.norm(vx_vy_comp, axis=0)+1e-5)*5
        plt.plot([-radar_point[1,:], -radar_point[1,:]-vx_vy_comp[1,:]], [radar_point[0,:], radar_point[0,:]+vx_vy_comp[0,:]], 'k-', color="r", linewidth = 0.5)
        
        for data in annotations[token_index]["annotations"]:
            if not (data["label"] in class_to_color):
                continue
            color = class_to_color[data["label"]][::-1]
            radar_indexes = data["radar_indexes"]
            corner = data["bev_box"]
            data["distance"] = None
            for i_corner in range(4):
                plt.plot([-corner[1][i_corner], -corner[1][(i_corner+1)%4]], [corner[0][i_corner], corner[0][(i_corner+1)%4]], 'k-', c = np.concatenate([color, [1]]), linewidth = 0.7)  
            if len(radar_indexes) > 0:
                np_points = np.array(radar_point[:,radar_indexes])
                dists = np.linalg.norm(np_points, axis=0)
                if False:
                    dist = dists.min()
                else:
                    dist = dists.mean()
                data["distance"] = dist
                gt_dist = np.linalg.norm(data["bev_box"])
                distance_errors.append(gt_dist - data["distance"])

                plt.scatter(-radar_point[1,radar_indexes], radar_point[0,radar_indexes], c = [color], s = scatter_size)

        bev_im_buf = io.BytesIO()
        #plt.legend(handles=legends, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
        plt.savefig(bev_im_buf, format='jpg', bbox_inches='tight')
        bev_im = cv2.imdecode(np.frombuffer(bev_im_buf.getvalue(), dtype=np.uint8), 1)

        cam_img = put_bbox_into_image(annotations[token_index], radar_in_image=radar_in_image[token_index], rcs_colors = rcs_colors)
        asp = cam_img.shape[0]/bev_im.shape[0]
        bev_im = cv2.resize(bev_im, dsize=(round(asp*bev_im.shape[1]), cam_img.shape[0]))
        out_img = cv2.hconcat([cam_img, bev_im])
        cv2.imwrite(os.path.join(save_dir, str(token_index).zfill(4)+".jpg"), out_img)
        
        out_img = cv2.resize(out_img, out_vid_shape)
        out_vid.write(out_img)

    out_vid.release()
    print("MAE: {}, std: {}".format(np.abs(distance_errors).mean(), np.std(np.abs(distance_errors))))