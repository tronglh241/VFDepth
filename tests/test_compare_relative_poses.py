import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())
from models.camera import Fisheye, PinHole  # noqa: E402
from models.vf_depth import VFDepth  # noqa: E402

sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth')))
sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth', 'external', 'packnet_sfm')))
sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth', 'external', 'dgp')))
sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth', 'external', 'monodepth2')))
from VFDepth import utils  # noqa: E402
from VFDepth.models import VFDepthAlgo  # noqa: E402
from VFDepth.models.geometry.geometry_util import vec_to_matrix  # noqa: E402
from VFDepth.network.blocks import pack_cam_feat, unpack_cam_feat  # noqa: E402
from VFDepth.utils.misc import _NUSC_CAM_LIST, get_relcam  # noqa: E402

_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']


if __name__ == '__main__':
    cfg = utils.get_config('VFDepth/configs/nuscenes/nusc_surround_fusion.yaml', mode='train')

    org_model = VFDepthAlgo(cfg, 0)
    org_model.load_weights()
    org_model.set_val()
    model = VFDepth()
    weight = torch.load('checkpoints/weights/vfdepth.pt')
    model.load_state_dict(weight)
    model.cuda()
    model.eval()

    # compare weights
    for net_type in ['pose_net', 'depth_net']:
        for module in ['encoder', 'conv1x1', 'fusion_net', 'decoder']:
            if net_type == 'pose_net' and module == 'decoder':
                org_module = 'pose_decoder'
            else:
                org_module = module

            for org_param, param in zip(
                eval(f'org_model.models["{net_type}"].{org_module}').parameters(),
                eval(f'model.{net_type}.{module}').parameters(),
            ):
                assert torch.all(abs(org_param - param) < 1e-8)

    org_pose_net = org_model.models['pose_net']
    pose_net = model.pose_net

    dataset = org_model.train_dataloader()
    inputs = next(iter(dataset))

    for key, ipt in inputs.items():
        if key not in _NO_DEVICE_KEYS:
            if 'context' in key:
                inputs[key] = [ipt[k].float().to(0) for k in range(len(inputs[key]))]
            else:
                inputs[key] = ipt.float().to(0)
    inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])

    with torch.no_grad():
        outputs = org_model.estimate_vfdepth(inputs)
        cur_to_prev_poses, cur_to_next_poses, depth_maps = model(
            prev_image=inputs['color_aug', -1, 0],
            cur_image=inputs['color_aug', 0, 0],
            next_image=inputs['color_aug', 1, 0],
            mask=inputs['mask'],
            intrinsic=inputs['K', 0],
            extrinsic=inputs['extrinsics_inv'],
            ref_extrinsic=inputs['extrinsics_inv'][:, 0:1],
            ref_inv_extrinsic=inputs['extrinsics'][:, 0:1],
            inv_extrinsic=inputs['extrinsics'],
            distortion=None,
        )

        for cam in range(org_model.num_cams):
            org_rel_pose_dict = org_model.pose.compute_relative_cam_poses(inputs, outputs, cam)
            rel_pose_dict = model.compute_relative_poses(
                cam_cur_to_prev_pose=cur_to_prev_poses[:, cam],
                cam_cur_to_next_pose=cur_to_next_poses[:, cam],
                cam_inv_extrinsic=inputs['extrinsics'][:, cam],
                extrinsic=inputs['extrinsics_inv'],
                neighbor_cam_indices=org_model.rel_cam_list[cam],
            )

            for k in org_rel_pose_dict:
                assert org_rel_pose_dict[k].shape == rel_pose_dict[k].shape
                assert torch.all(abs(org_rel_pose_dict[k] - rel_pose_dict[k]) < 1e-9)
