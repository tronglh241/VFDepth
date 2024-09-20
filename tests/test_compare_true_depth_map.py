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

    outputs = org_model.estimate_vfdepth(inputs)
    depth_maps = torch.stack([outputs['cam', cam]['disp', 0] for cam in range(6)], dim=1)

    true_depth_maps = model.compute_true_depth_maps(
        depth_maps=depth_maps,
        intrinsic=inputs['K', 0],
        max_depth=org_model.max_depth,
        min_depth=org_model.min_depth,
        focal_length_scale=org_model.focal_length_scale,
    )

    for cam in range(6):
        assert true_depth_maps[:, cam].shape == outputs['cam', cam]['depth', 0].shape
        assert torch.all(abs(true_depth_maps[:, cam] - outputs['cam', cam]['depth', 0]) < 1e-9)
