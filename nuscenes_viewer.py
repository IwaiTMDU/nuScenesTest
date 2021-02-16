from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, PointCloud
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.patches as mpatches
from tqdm import tqdm
import open3d as o3d
import cv2
import io
from PIL import Image
from pyquaternion import Quaternion
import copy
from nuscenes.utils.geometry_utils import BoxVisibility

version = "v1.0-mini"
dataroot = "v1.0-mini"
box_visibility = BoxVisibility.ANY
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

def put_bbox_into_image(annotation, radar_in_image = None, selected_point_ids = None, rcs_colors = None, radar_points_distance = None):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1.0

    image = cv2.imread(annotation["image_file"])
    
    for data in annotation["annotations"]:
        if not (data["label"] in class_to_color):
            continue
        (x1, y1, x2, y2) = data["box"].astype(np.int32)
        color =class_to_color[data["label"]] * 255
        put_class_label = data["label"].split('.')[-1]
        text = put_class_label
        if data["distance"] is not None:
            text += "[{}m]".format(round(data["distance"], 1))
        else: # bbox has no radar points
            no_point_color = (0, 0, 0)
            cv2.rectangle(image,(x1, y1), (x2, y2), no_point_color, 12)
        (retval,baseLine) = cv2.getTextSize(text, font, fontScale,1)
        textOrg = int(x1), int(y1)

        cv2.rectangle(image,(x1, y1), (x2, y2), color,2) # bbox
        cv2.rectangle(image, (textOrg[0] - 1,textOrg[1]+baseLine - 1), (textOrg[0]+retval[0] + 1, textOrg[1]-retval[1] - 1), color, -1) # text box
        cv2.putText(image, text, textOrg, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (1,1,1), 1) # text
        
    image_rcs = image.copy()
    image_range = image.copy()
    max_distance = 100
    radar_points_distance = np.clip(radar_points_distance, 0, max_distance)

    if (radar_in_image is not None) and (rcs_colors is not None):
        for point_index in range(radar_in_image.shape[1]):
            point = radar_in_image[:,point_index]
            u, v = int(point[0]), int(point[1])
            point_color = rcs_colors[token_index][point_index]
            point_color = (int(point_color[0]), int(point_color[1]), int(point_color[2]))
            range_hue = 120*(radar_points_distance[point_index]/max_distance)
            range_hsv = 255*np.ones(3).astype(np.uint8)
            range_hsv[0] = range_hue
            range_bgr = cv2.cvtColor(np.array([[range_hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR).reshape(3)

            if ((0 <= u <= image.shape[1]) and (0 <= v <= image.shape[0])): # point is in the image
                if point_index in selected_point_ids:
                    cv2.circle(image_rcs, (u, v), 12, color = point_color, thickness = -1, lineType = cv2.LINE_8)    
                    cv2.circle(image_rcs, (u, v), 6, color = [0, 0, 0, 0], thickness = -1, lineType = cv2.LINE_8)

                    cv2.circle(image_range, (u, v), 12, color = (int(range_bgr[0]), int(range_bgr[1]), int(range_bgr[2])), thickness = -1, lineType = cv2.LINE_8)    
                    cv2.circle(image_range, (u, v), 6, color = [0, 0, 0, 0], thickness = -1, lineType = cv2.LINE_8)
                else:
                    cv2.circle(image_rcs, (u, v), 6, color = point_color, thickness = -1, lineType = cv2.LINE_8)
                    cv2.circle(image_range, (u, v), 6, color = (int(range_bgr[0]), int(range_bgr[1]), int(range_bgr[2])), thickness = -1, lineType = cv2.LINE_8)
    
    output_image = cv2.vconcat([image_rcs, image_range])

    return output_image

def get_annotation_bbox(nusc, tokens):
    annotations = []
    for token_index in tqdm(range(len(tokens))):
        token = tokens[token_index]
        sample = nusc.get('sample', token)
        camera_rec = nusc.get('sample_data', sample['data']["CAM_FRONT"])
        radar_rec = nusc.get('sample_data', sample['data']["RADAR_FRONT"])
        camera_pos = nusc.get('calibrated_sensor', camera_rec['calibrated_sensor_token'])['translation']
        radar_pos = nusc.get('calibrated_sensor', radar_rec['calibrated_sensor_token'])['translation']
        camera_to_radar_pos = np.array(camera_pos) - np.array(radar_pos)
        _, boxes, camera_intrinsic = nusc.get_sample_data(camera_rec['token'], box_vis_level=box_visibility)

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
        #rcs_positive = np.power(10, radar_meta_data[token_index][2,:]/20.0)
        #max_rcs = np.max(rcs_positive)
        max_rcs = np.max(radar_meta_data[token_index][2,:])
        min_rcs = np.min(radar_meta_data[token_index][2,:])
        rcs_hue = (120*((radar_meta_data[token_index][2,:] - min_rcs)/(max_rcs - min_rcs))).astype(np.uint8)
        rcs_hsv = 255*np.ones((rcs_hue.shape[0], 3)).astype(np.uint8)
        rcs_hsv[:,0] = rcs_hue
        
        rcs_colors = np.array([cv2.cvtColor(np.array([[rcs_hsv[point_idx]]], dtype=np.uint8), cv2.COLOR_HSV2RGB) for point_idx in range(rcs_hsv.shape[0])])
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


def get_class_priority(label):
    '''
    classes
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
    '''

    if 'bicycle' in label:
        return 2
    elif 'car' in label:
        return 1
    else:
        return 0


def compare_bbox_area(bbox1, bbox2):
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    return 0 if area1 > area2 else 1


def check_radar_in_2dbbox2(tokens, annotations, radar_in_image):
    for token_index in tqdm(range(len(tokens))):
        token_annotations = annotations[token_index]["annotations"]
        for ann_index in range(len(token_annotations)):
            token_annotations[ann_index]["radar_indexes"] = []

        for point_index in range(radar_in_image[token_index].shape[1]):
            point = radar_in_image[token_index][:,point_index]
            u, v = int(point[0]), int(point[1])
            
            bbox_candidate = [] # pointがbbox内にあるbbox
            for ann_index, annotation in enumerate(token_annotations):
                box = annotation["box"]
                if((box[0] <= u <= box[2]) and (box[1] <= v <= box[3])): # bbox内にあるとき
                    bbox_candidate.append(ann_index) # annotationのデータを一旦入れておく
            if len(bbox_candidate) == 1:
                # bboxが一つならそれをpoint_indexに追加 
                token_annotations[bbox_candidate[0]]["radar_indexes"].append(point_index)
            
            elif len(bbox_candidate) > 1:
                high_prio_index = bbox_candidate[0]
                for bbox_cand in bbox_candidate[1:]:
                    high_prio_ann = token_annotations[high_prio_index]
                    cand_ann = token_annotations[bbox_cand]
                    if get_class_priority(high_prio_ann["label"]) == get_class_priority(cand_ann["label"]):# 同じ優先度
                        if compare_bbox_area(high_prio_ann["box"], cand_ann["box"]) != 0:
                            high_prio_index = bbox_cand
                        
                    elif get_class_priority(high_prio_ann["label"]) > get_class_priority(cand_ann["label"]):# 自転車 < 車 < トラック
                        high_prio_index = bbox_cand

                token_annotations[high_prio_index]["radar_indexes"].append(point_index)
            
    return annotations


if __name__ == "__main__":
    nusc = NuScenes(version = version, dataroot = dataroot, verbose=False)
    scene_num = len(nusc.scene)
    sample_tokens = {}
    prog = 0
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)
    distance_errors = []
    distance_labels = []
    distance_centers = []
    distance_files = []
    distance_dist=[]
    distance_phd0 = []
    error_of_1file = []
    best_files = []
    token_scene = []
    scene_videos = {}

    # Assign color
    class_to_color['bg'] = np.zeros(3)
    for class_id, class_name in enumerate(class_names):
        class_color_hsv = 255*np.ones(3).astype(np.uint8)
        class_color_hsv[0] = np.uint8((float(class_id) / len(class_names))**2 * 120)
        class_to_color[class_name] = cv2.cvtColor(np.array([[class_color_hsv]], dtype=np.uint8), cv2.COLOR_HSV2BGR)/255.0
        class_to_color[class_name] = class_to_color[class_name].reshape(3)
    
    # Get all tokens
    for scene_index in range(scene_num):
        first_sample_token = nusc.scene[scene_index]['first_sample_token']
        nbr_samples = nusc.scene[scene_index]['nbr_samples']
        curr_sample = nusc.get('sample', first_sample_token)

        for _ in range(nbr_samples):
            sample_tokens[prog] = curr_sample['token']
            token_scene.append(nusc.scene[scene_index]["name"])
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

    out_vid_shape = (1440, 810)
    out_vid = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 0.5, out_vid_shape)
    for scene_index in range(scene_num):
        scene_name = nusc.scene[scene_index]["name"]
        scene_videos[scene_name] = cv2.VideoWriter("{}.mp4".format(scene_name), cv2.VideoWriter_fourcc(*'mp4v'), 2, out_vid_shape)

    for token_index in tqdm(range(len(sample_tokens))):
        radar_point = radar_points[token_index]

        plt.clf()
        plt.xlabel("y [m]")
        plt.ylabel("x(forward) [m]")
        vert_lim = 100
        horiz_lim = 30
        near_max_angle = 60
        middle_max_angle = 45
        long_max_angle = 9
        plt.xlim([-horiz_lim, horiz_lim])
        plt.ylim([0, vert_lim])
        plt.xticks(range(-horiz_lim, horiz_lim+1, 10))
        plt.yticks(range(0, vert_lim+1, 10))
        plt.grid(which = "major", axis ="x", color = "black", alpha = 0.6, linewidth = 1 )
        plt.grid(which = "major", axis ="y", color = "black", alpha = 0.6, linewidth = 1 )
        plt.grid(which = "minor", axis ="x", color = "black", alpha = 0.6, linewidth = 1 )
        plt.grid(which = "minor", axis ="y", color = "black", alpha = 0.6, linewidth = 1 )
        long_range_fov = pat.Wedge(center = (0, 0), r = 250, theta1 = 90 - long_max_angle, theta2 = 90 + long_max_angle, color = "red", alpha = 0.08)
        middle_range_fov = pat.Wedge(center = (0, 0), r = 100, theta1 = 90 - middle_max_angle, theta2 = 90 + middle_max_angle, color = "green", alpha = 0.08)
        short_range_fov = pat.Wedge(center = (0, 0), r = 20, theta1 = 90 - near_max_angle, theta2 = 90 + near_max_angle, color = "blue", alpha = 0.08)
        ax = plt.axes()
        ax.add_patch(long_range_fov)
        ax.add_patch(middle_range_fov)
        ax.add_patch(short_range_fov)
        # plt.scatter(-radar_point[1,:], radar_point[0,:], c = "black", s = scatter_size)
        plt.scatter(-radar_point[1,:], radar_point[0,:], c = np.concatenate([np.array(rcs_colors[token_index])/255.0, np.ones((rcs_colors[token_index].shape[0], 1))], axis=1), s = scatter_size)

        vx_vy_comp = radar_meta_data[token_index][5:7,:]*4
        selected_point_ids = []
        point_dists = np.linalg.norm(radar_point[:2, :], axis=0)
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
                np_points = np.array(radar_point[:2,radar_indexes])
                dists = point_dists[radar_indexes]
                if True:
                    dist = dists.min()
                else:
                    dist = dists.mean()
                data["distance"] = dist
                selected_point_id = dists.argmin()
                selected_point = np_points[:,selected_point_id]
                selected_point_ids.append(radar_indexes[selected_point_id])
                gt_center = data["bev_box"].mean(axis=1)
                gt_dist = np.linalg.norm(gt_center)
                distance_errors.append(np.abs(gt_dist - dist))
                distance_labels.append(data["label"])
                distance_centers.append(gt_center)
                distance_files.append(token_index)
                distance_dist.append(dist)
                distance_phd0.append(radar_meta_data[token_index][12][radar_indexes])
                plt.plot([-gt_center[1], -selected_point[1]], [gt_center[0], selected_point[0]],'k-', c = [0, 0, 0.4, 1], linewidth = 1.5)
                
                #plt.scatter(-radar_point[1,radar_indexes], radar_point[0,radar_indexes], c = [color], s = scatter_size) # class color
        error_of_1file.append(np.mean(distance_errors))
        best_files.append(token_index)
        bev_im_buf = io.BytesIO()
        #plt.legend(handles=legends, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
        plt.savefig(bev_im_buf, format='jpg', bbox_inches='tight')
        bev_im = cv2.imdecode(np.frombuffer(bev_im_buf.getvalue(), dtype=np.uint8), 1)

        cam_img = put_bbox_into_image(annotations[token_index], radar_in_image=radar_in_image[token_index], selected_point_ids=selected_point_ids, rcs_colors = rcs_colors, radar_points_distance = point_dists)
        asp = cam_img.shape[0]/bev_im.shape[0]
        bev_im = cv2.resize(bev_im, dsize=(round(asp*bev_im.shape[1]), cam_img.shape[0]))
        out_img = cv2.hconcat([cam_img, bev_im])
        cv2.imwrite(os.path.join(save_dir, str(token_index).zfill(4)+".jpg"), out_img)
        
        out_img = cv2.resize(out_img, out_vid_shape)
        scene_videos[token_scene[token_index]].write(out_img)
        out_vid.write(out_img)

    for scene_index in range(scene_num):
        scene_name = nusc.scene[scene_index]["name"]
        scene_videos[scene_name].release()
    out_vid.release()
    
    print("MAE: {}, std: {}".format(np.mean(distance_errors), np.std(distance_errors)))

    dis_label = sorted(zip(error_of_1file, best_files))
    topk = 20
    best, files = zip(*dis_label)
    print("Best dist {}".format(best[:topk]))
    print("Best file {}".format(files[:topk]))

    dis_label = sorted(zip(distance_errors, distance_labels, distance_files, distance_centers, distance_dist, distance_phd0))
    worst, labels, files, centers, dists, phd0s = zip(*dis_label)
    topk = 10
    print("Worst error {}".format(worst[::-1][:topk]))
    print("Worst file {}".format(files[::-1][:topk]))
    print("Worst center {}".format(centers[::-1][:topk]))