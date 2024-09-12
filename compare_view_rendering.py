import torch
from torch.utils.data import DataLoader

import utils
from flame_based.models.view_renderer import ViewRenderer
from models import VFDepthAlgo
from utils.misc import _NUSC_CAM_LIST, get_relcam
_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']

if __name__ == '__main__':
    cfg = utils.get_config('./configs/nuscenes/nusc_surround_fusion.yaml', mode='train')

    org_model = VFDepthAlgo(cfg, 0)
    org_renderer = org_model.view_rendering
    renderer = ViewRenderer(org_model.height, org_model.width)

    dataset = org_model.train_dataloader()
    inputs = next(iter(dataset))

    for key, ipt in inputs.items():
        if key not in _NO_DEVICE_KEYS:
            if 'context' in key:
                inputs[key] = [ipt[k].float().to(0) for k in range(len(inputs[key]))]
            else:
                inputs[key] = ipt.float().to(0)

    outputs = org_model.estimate_vfdepth(inputs)
    for cam in range(org_model.num_cams):
        rel_pose_dict = org_model.pose.compute_relative_cam_poses(inputs, outputs, cam)
        renderer.forward(
            org_prev_image,
            org_image,
            org_next_image,
            mask,
            intrinsic,
            inv_intrinsic,
            true_depth_map,
            prev_to_cur_pose,
            cur_to_next_pose,
            cam_index,
            neighbor_cam_indices,
            rel_pose_dict,
        )
        org_model.view_rendering(inputs, outputs, cam, rel_pose_dict)
