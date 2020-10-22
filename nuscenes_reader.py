from nuscenes.nuscenes import NuScenes as nu
nusc = nu(version = "v1.0-mini", dataroot = "v1.0-mini", verbose=False)
sample = nusc.get("sample", nusc.scene[0]['first_sample_token'])
sensor = "CAM_FRONT"
sensor_data = nusc.get('sample_data', sample['data'][sensor])
