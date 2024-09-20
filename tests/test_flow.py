import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())
from losses.loss_computation_wrapper import \
    LossComputationWrapper  # noqa: E402
from losses.multi_cam_loss import MultiCamLoss  # noqa: E402
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

    loss_fn = MultiCamLoss(
        disparity_smoothness=org_model.disparity_smoothness,
        spatio_coeff=org_model.spatio_coeff,
        spatio_tempo_coeff=org_model.spatio_tempo_coeff,
    )
    loss_computation_wrapper = LossComputationWrapper(
        model=model,
        loss_fn=loss_fn,
        max_depth=org_model.max_depth,
        min_depth=org_model.min_depth,
        focal_length_scale=org_model.focal_length_scale,
        neighbor_cam_indices_map=org_model.rel_cam_list,
    )
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

    dataset = org_model.train_dataloader()
    inputs = next(iter(dataset))

    with torch.no_grad():
        outputs = org_model.process_batch(inputs, 0)

        cur_to_prev_poses, cur_to_next_poses, depth_maps = model.forward(
            prev_image=inputs['color_aug', -1, 0],
            cur_image=inputs['color_aug', 0, 0],
            next_image=inputs['color_aug', 1, 0],
            mask=inputs['mask'],
            intrinsic=inputs['K', 0],
            extrinsic=inputs['extrinsics_inv'],
            ref_extrinsic=inputs['extrinsics_inv'][:, 0:1],
            ref_inv_extrinsic=None,
            inv_intrinsic=None,
            inv_extrinsic=None,
            distortion=None,
        )
        loss = loss_computation_wrapper(
            org_prev_images=inputs['color', -1, 0],
            org_cur_images=inputs['color', 0, 0],
            org_next_images=inputs['color', 1, 0],
            masks=inputs['mask'],
            depth_maps=depth_maps,
            intrinsics=inputs['K', 0],
            extrinsics=inputs['extrinsics_inv'],
            cur_to_prev_poses=cur_to_prev_poses,
            cur_to_next_poses=cur_to_next_poses,
            distortions=None,
        )
        breakpoint()
