import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, PointCloud
from nuscenes.utils.geometry_utils import view_points
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.patches as mpatches
from tqdm import tqdm
import io
from PIL import Image
from pyquaternion import Quaternion
import copy
from nuscenes.utils.geometry_utils import BoxVisibility
import time

from nuscenes_viewer import get_annotation_bbox
from camera_radar_fusion import camera_to_bev

version = "v1.0-mini"
dataroot = "v1.0-mini"
box_visibility = BoxVisibility.ANY
object_heights = {
    'vehicle.bicycle':1.279, 
    'vehicle.bus.bendy':3.625, 
    'vehicle.bus.rigid':4.016, 
    'vehicle.car':1.8034, 
    'vehicle.construction':2.355, 
    'vehicle.motorcycle':1.460, 
    'vehicle.trailer':3.410, 
    'vehicle.truck':2.791}
class_to_color = {}
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

def put_bbox_into_image(preds, selected_point_ids, radar_points_distance, img):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1.0
    
    image = img.copy()
    for data in preds:
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

    return image


def get_max_iou_ann_idx(a, b):
    # 矩形aの面積a_areaを計算
    a_area = (a[2] - a[0] + 1) \
             * (a[3] - a[1] + 1)
    # bに含まれる矩形のそれぞれの面積b_areaを計算
    # shape=(N,)のnumpy配列。Nは矩形の数
    b_area = (b[:,2] - b[:,0] + 1) \
             * (b[:,3] - b[:,1] + 1)
    
    # aとbの矩形の共通部分(intersection)の面積を計算するために、
    # N個のbについて、aとの共通部分のxmin, ymin, xmax, ymaxを一気に計算
    abx_mn = np.maximum(a[0], b[:,0]) # xmin
    aby_mn = np.maximum(a[1], b[:,1]) # ymin
    abx_mx = np.minimum(a[2], b[:,2]) # xmax
    aby_mx = np.minimum(a[3], b[:,3]) # ymax
    # 共通部分の矩形の幅を計算。共通部分が無ければ0
    w = np.maximum(0, abx_mx - abx_mn + 1)
    # 共通部分の矩形の高さを計算。共通部分が無ければ0
    h = np.maximum(0, aby_mx - aby_mn + 1)
    # 共通部分の面積を計算。共通部分が無ければ0
    intersect = w*h
    
    # N個のbについて、aとのIoUを一気に計算
    iou = intersect / (a_area + b_area - intersect)
    max_iou_idx = np.argmax(iou)
    
    return max_iou_idx


def inference(img, net, output_layers, coco_classes):
    pick_classes = {
        'bicycle':'vehicle.bicycle',
        'car':'vehicle.car',
        'motorbike':'vehicle.motorcycle',
        'bus':'vehicle.bus.rigid',
        'truck':'vehicle.truck'
    }

    preds = []
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=img.shape[:2], mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            objness = scores[class_id]
            class_name = coco_classes[class_id]
            if class_name in pick_classes.keys():
                nusc_class_name = pick_classes[class_name]
                if objness > 0.8:
                    cx = detect[0]
                    cy = detect[1]
                    w = detect[2]
                    h = detect[3]
                    left = cx - 0.5 * w
                    top = cy - 0.5 * h
                    right = left + w
                    bottom = top + h
                    preds.append({"label":nusc_class_name, "box":np.array([left, top, right, bottom])})

    return preds


if __name__ == '__main__':
    nusc = NuScenes(version = version, dataroot = dataroot, verbose=False)
    scene_num = len(nusc.scene)
    sample_tokens = {}
    prog = 0
    save_dir = "result"
    os.makedirs(save_dir, exist_ok=True)
    token_scene = []
    scene_videos = {}
    distance_errors = {}
    gt_error_data = {}
    found_bboxes = []
    for class_name in class_names:
        distance_errors[class_name] = []
        gt_error_data[class_name] = []

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

    annotations, _ = get_annotation_bbox(nusc, sample_tokens)

    # load yolov3
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    coco_classes = []
    with open("coco.names", "r") as f:
        coco_classes = [line.strip() for line in f.readlines()]
    
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    width = 416
    height = 416

    out_vid_shape = (1440, 810)
    out_vid = cv2.VideoWriter("result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 0.5, out_vid_shape)
    time_calc_distance = []
    for scene_index in range(scene_num):
        scene_name = nusc.scene[scene_index]["name"]
        scene_videos[scene_name] = cv2.VideoWriter("{}.mp4".format(scene_name), cv2.VideoWriter_fourcc(*'mp4v'), 2, out_vid_shape)

    for token_index in tqdm(range(len(sample_tokens))):
        img = cv2.imread(annotations[token_index]["image_file"])
        img_resized = cv2.resize(img, (width, height))
        preds = inference(img_resized, net, output_layers, coco_classes)
        ann_bboxes = np.array([annotation["box"] for annotation in annotations[token_index]["annotations"]])
        
        max_iou_anns = []
        max_iou_idxs = []
        for pred in preds:
            pred["box"] = (pred["box"] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])).astype(np.int)
            max_iou_idx = get_max_iou_ann_idx(pred["box"], ann_bboxes)
            max_iou_idxs.append(max_iou_idx)
            max_iou_anns.append(annotations[token_index]["annotations"][max_iou_idx])
        max_iou_idxs = list(set(max_iou_idxs))
        found_bboxes.append([len(max_iou_idxs), ann_bboxes.shape[0]])

        selected_point_ids, selected_points, distances = camera_to_bev(preds, radar_point = None, nusc = nusc, token = sample_tokens[token_index], heights = object_heights)
        
        for i, pred in enumerate(preds):
            pred["distance"] = distances[i]

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

        for i, pred in enumerate(preds):
            color = class_to_color[max_iou_anns[i]["label"]][::-1]
            corner = max_iou_anns[i]["bev_box"]
            gt_center = max_iou_anns[i]["bev_box"].mean(axis=1)
            selected_point = selected_points[i]
            for i_corner in range(4): # plot bev bbox
                plt.plot([-corner[1][i_corner], -corner[1][(i_corner+1)%4]], [corner[0][i_corner], corner[0][(i_corner+1)%4]], 'k-', c = np.concatenate([color, [1]]), linewidth = 0.7)
            plt.plot([-gt_center[1], -selected_point[1]], [gt_center[0], selected_point[0]],'k-', c = [0, 0, 0.4, 1], linewidth = 1.5)
        
            point_distance = np.linalg.norm(selected_point)
            gt_distance = np.linalg.norm(gt_center[:2])
            distance_error = np.abs(point_distance - gt_distance)
            distance_errors[max_iou_anns[i]["label"]].append(distance_error)
            gt_error_data[max_iou_anns[i]["label"]].append(gt_distance)

        bev_im_buf = io.BytesIO()
        plt.savefig(bev_im_buf, format='jpg', bbox_inches='tight')
        bev_im = cv2.imdecode(np.frombuffer(bev_im_buf.getvalue(), dtype=np.uint8), 1)

        cam_img = put_bbox_into_image(preds, selected_point_ids, distances, img)
        asp = cam_img.shape[0]/bev_im.shape[0]
        bev_im = cv2.resize(bev_im, dsize=(round(asp*bev_im.shape[1]), cam_img.shape[0]))
        out_img = cv2.hconcat([cam_img, bev_im])
        cv2.imwrite(os.path.join(save_dir, str(token_index).zfill(4)+".jpg"), out_img)
        scene_videos[token_scene[token_index]].write(out_img)
        out_vid.write(out_img)

    distance_error_array = []
    for distance_error in distance_errors.values():
        distance_error_array += distance_error

    found_bboxes_cnt = np.array(found_bboxes).sum(axis=0)
    print("Found bbox {} / {}".format(found_bboxes_cnt[0], found_bboxes_cnt[1]))
    print("Distance error mean : {}, std : {}".format(np.mean(distance_error_array), np.std(distance_error_array)))

