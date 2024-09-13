import torch

import utils
from flame_based.models.view_renderer import ViewRenderer
from models import VFDepthAlgo
from utils.misc import _NUSC_CAM_LIST, get_relcam
_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']

if __name__ == '__main__':
    cfg = utils.get_config('./configs/nuscenes/nusc_surround_fusion.yaml', mode='train')

    org_model = VFDepthAlgo(cfg, 0)
    org_renderer = org_model.view_rendering
    renderer = ViewRenderer()

    dataset = org_model.train_dataloader()
    inputs = next(iter(dataset))

    for key, ipt in inputs.items():
        if key not in _NO_DEVICE_KEYS:
            if 'context' in key:
                inputs[key] = [ipt[k].float().to(0) for k in range(len(inputs[key]))]
            else:
                inputs[key] = ipt.float().to(0)

    outputs = org_model.estimate_vfdepth(inputs)
    for cam_index in range(org_model.num_cams):
        rel_pose_dict = org_model.pose.compute_relative_cam_poses(inputs, outputs, cam_index)
        org_prev_image = inputs['color', -1, 0]
        org_image = inputs['color', 0, 0]
        org_next_image = inputs['color', 1, 0]
        mask = inputs['mask']
        intrinsic = inputs['K', 0]
        extrinsic = inputs['extrinsics_inv']
        inv_intrinsic = inputs['inv_K', 0]
        true_depth_map = torch.stack([outputs['cam', cam]['depth', 0] for cam in range(6)], dim=1)
        assert true_depth_map.shape == (
            org_model.batch_size, org_model.num_cams, 1, org_model.height, org_model.width), true_depth_map.shape
        prev_to_cur_pose = torch.stack([outputs['cam', cam]['cam_T_cam', 0, -1] for cam in range(6)], dim=1)
        next_to_cur_pose = torch.stack([outputs['cam', cam]['cam_T_cam', 0, 1] for cam in range(6)], dim=1)
        org_model.view_rendering(inputs, outputs, cam_index, rel_pose_dict)
        warped_views = renderer.forward(
            org_prev_image,
            org_image,
            org_next_image,
            mask,
            intrinsic,
            inv_intrinsic,
            true_depth_map,
            prev_to_cur_pose,
            next_to_cur_pose,
            cam_index,
            get_relcam(_NUSC_CAM_LIST)[cam_index],
            rel_pose_dict,
            extrinsic,
        )
        print(torch.all(abs(outputs['cam', cam_index]['color', -1, 0] - warped_views['color', -1])) < 1e-9)
        print(torch.all(abs(outputs['cam', cam_index]['color', 1, 0] - warped_views['color', 1])) < 1e-9)
        print(torch.all(abs(outputs['cam', cam_index]['color_mask', -1, 0] - warped_views['color_mask', -1])) < 1e-9)
        print(torch.all(abs(outputs['cam', cam_index]['color_mask', 1, 0] - warped_views['color_mask', 1])) < 1e-9)

        print(torch.all(abs(outputs['cam', cam_index]['overlap', -1, 0] - warped_views['overlap', -1])) < 1e-9)
        print(torch.all(abs(outputs['cam', cam_index]['overlap', 1, 0] - warped_views['overlap', 1])) < 1e-9)
        print(torch.all(abs(outputs['cam', cam_index]['overlap_mask', -1, 0] - warped_views['overlap_mask', -1])) < 1e-9)
        print(torch.all(abs(outputs['cam', cam_index]['overlap_mask', 1, 0] - warped_views['overlap_mask', 1])) < 1e-9)
