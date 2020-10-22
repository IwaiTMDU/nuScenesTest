from nuscenes.nuscenes import NuScenes as nu
import os
from PIL import Image

dataroot = "v1.0-mini"
scene_index = 0
nusc = nu(version = "v1.0-mini", dataroot = dataroot, verbose=False)
sample = nusc.get("sample", nusc.scene[scene_index]['first_sample_token'])
sensor = "CAM_FRONT"

# Camera
imgs = []
for _ in range(nusc.scene[scene_index]["nbr_samples"]-1):
	sensor_data = nusc.get('sample_data', sample['data'][sensor])
	img = Image.open(os.path.join(dataroot,sensor_data['filename']))
	imgs.append(img)
	sample = nusc.get("sample", sample['next'])

imgs[0].save('nuscenes_imgs.gif', save_all=True, append_images=imgs[1:])

# Radar
sample = nusc.get("sample", nusc.scene[scene_index]['first_sample_token'])
sensor = "RADAR_FRONT"
for _ in range(nusc.scene[scene_index]["nbr_samples"]-1):
	sensor_data = nusc.get('sample_data', sample['data'][sensor])
	sample = nusc.get("sample", sample['next'])