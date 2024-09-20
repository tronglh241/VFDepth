import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())
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

    dataset = org_model.val_dataloader()
    inputs = next(iter(dataset))

    for key, ipt in inputs.items():
        if key not in _NO_DEVICE_KEYS:
            if 'context' in key:
                inputs[key] = [ipt[k].float().to(0) for k in range(len(inputs[key]))]
            else:
                inputs[key] = ipt.float().to(0)
    inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])

    loss_fn = MultiCamLoss(
        disparity_smoothness=org_model.disparity_smoothness,
        spatio_coeff=org_model.spatio_coeff,
        spatio_tempo_coeff=org_model.spatio_tempo_coeff,
    )

    outputs = org_model.estimate_vfdepth(inputs)

    for cam in range(org_model.num_cams):
        print(f'Cam #{cam}')
        org_model.pred_cam_imgs(inputs, outputs, cam)
        # org_cam_loss, org_loss_dict = org_model.losses(inputs, outputs, cam)
        # loss_dict = {}
        # cam_loss = 0.  # loss across the multi-scale
        # target_view = outputs[('cam', cam)]
        # for scale in org_model.losses.scales:
        #     kargs = {
        #         'cam': cam,
        #         'scale': scale,
        #         'ref_mask': inputs['mask'][:, cam, ...]
        #     }

        #     reprojection_loss = org_model.losses.compute_reproj_loss(inputs, target_view, **kargs)
        #     smooth_loss = org_model.losses.compute_smooth_loss(inputs, target_view, **kargs)
        #     spatio_loss = org_model.losses.compute_spatio_loss(inputs, target_view, **kargs)

        #     kargs['reproj_loss_mask'] = target_view[('reproj_mask', scale)]
        #     spatio_tempo_loss = org_model.losses.compute_spatio_tempo_loss(inputs, target_view, **kargs)

        #     # pose consistency loss
        #     if org_model.losses.pose_model == 'fsm' and cam != 0:
        #         pose_loss = org_model.losses.compute_pose_con_loss(inputs, outputs, **kargs)
        #     else:
        #         pose_loss = 0

        #     cam_loss += reprojection_loss
        #     cam_loss += org_model.losses.disparity_smoothness * smooth_loss / (2 ** scale)
        #     cam_loss += org_model.losses.spatio_coeff * spatio_loss + org_model.losses.spatio_tempo_coeff * spatio_tempo_loss
        #     cam_loss += org_model.losses.pose_loss_coeff * pose_loss

        #     ##########################
        #     # for logger
        #     ##########################
        #     if scale == 0:
        #         loss_dict['reproj_loss'] = reprojection_loss.item()
        #         loss_dict['spatio_loss'] = spatio_loss.item()
        #         loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item()
        #         loss_dict['smooth'] = smooth_loss.item()
        #         if org_model.losses.pose_model == 'fsm' and cam != 0:
        #             loss_dict['pose'] = pose_loss.item()

        #         # log statistics
        #         org_model.losses.get_logs(loss_dict, target_view, cam)

        # cam_loss /= len(org_model.losses.scales)

        cam_target_view = {}

        for key in outputs['cam', cam].keys():
            if key[0].startswith('color') or key[0].startswith('overlap'):
                cam_target_view[key[:2]] = outputs['cam', cam][key]

        # org_cam_org_prev_image = inputs['color', -1, 0][:, cam].clone()
        # org_cam_org_image = inputs['color', 0, 0][:, cam].clone()
        # org_cam_org_next_image = inputs['color', 1, 0][:, cam].clone()
        # org_cam_target_view = {k: v.clone() for k, v in cam_target_view.items()}
        # org_cam_depth_map = outputs['cam', cam]['disp', 0].clone()
        # org_cam_mask = inputs['mask'][:, cam].clone()
        org_cam_loss, org_loss_dict = loss_fn(
            cam_org_prev_image=inputs['color', -1, 0][:, cam],
            cam_org_image=inputs['color', 0, 0][:, cam],
            cam_org_next_image=inputs['color', 1, 0][:, cam],
            cam_target_view=cam_target_view,
            cam_depth_map=outputs['cam', cam]['disp', 0],
            cam_mask=inputs['mask'][:, cam],
        )

        cam_org_prev_image = inputs['color', -1, 0][:, cam]
        cam_org_image = inputs['color', 0, 0][:, cam]
        cam_org_next_image = inputs['color', 1, 0][:, cam]
        cam_target_view = cam_target_view
        cam_depth_map = outputs['cam', cam]['disp', 0]
        cam_mask = inputs['mask'][:, cam]

        # for org_tensor, tensor in zip(
        #     [
        #         org_cam_org_prev_image,
        #         org_cam_org_image,
        #         org_cam_org_next_image,
        #         # org_cam_target_view,
        #         org_cam_depth_map,
        #         org_cam_mask,
        #     ],
        #     [
        #         cam_org_prev_image,
        #         cam_org_image,
        #         cam_org_next_image,
        #         # cam_target_view,
        #         cam_depth_map,
        #         cam_mask,
        #     ],
        # ):
        #     assert org_tensor.shape == tensor.shape
        #     assert torch.all(abs(org_tensor - tensor) < 1e-9), abs(org_tensor - tensor)

        # for k in org_cam_target_view:
        #     assert org_cam_target_view[k].shape == cam_target_view[k].shape
        #     assert torch.all(abs(org_cam_target_view[k] - cam_target_view[k]) < 1e-9), abs(org_cam_target_view[k] - cam_target_view[k]).max()

        cam_loss, loss_dict = loss_fn(
            cam_org_prev_image=cam_org_prev_image,
            cam_org_image=cam_org_image,
            cam_org_next_image=cam_org_next_image,
            cam_target_view=cam_target_view,
            cam_depth_map=cam_depth_map,
            cam_mask=cam_mask,
        )

        print(org_cam_loss - cam_loss)
        for k in loss_dict:
            print(k, org_loss_dict[k] - loss_dict[k])
