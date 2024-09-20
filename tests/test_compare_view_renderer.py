import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())
from models.camera import Fisheye, PinHole  # noqa: E402
from models.vf_depth import VFDepth  # noqa: E402
from models.view_renderer import ViewRenderer  # noqa: E402

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
        true_depth_maps = torch.stack([outputs['cam', cam]['depth', 0] for cam in range(6)], dim=1)

        view_renderer = ViewRenderer()

        for cam in range(org_model.num_cams):
            org_rel_pose_dict = org_model.pose.compute_relative_cam_poses(inputs, outputs, cam)
            # source_scale = 0

            # # ref inputs
            # ref_color = inputs['color', 0, source_scale][:, cam, ...]
            # ref_mask = inputs['mask'][:, cam, ...]
            # ref_K = inputs[('K', source_scale)][:, cam, ...]
            # ref_invK = inputs[('inv_K', source_scale)][:, cam, ...]

            # # output
            # target_view = outputs[('cam', cam)]

            # for scale in org_model.view_rendering.scales:
            #     ref_depth = target_view[('depth', scale)]
            #     for frame_id in org_model.view_rendering.frame_ids[1:]:
            #         # for temporal learning
            #         T = target_view[('cam_T_cam', 0, frame_id)]
            #         src_color = inputs['color', frame_id, source_scale][:, cam, ...]
            #         src_mask = inputs['mask'][:, cam, ...]
            #         warped_img, warped_mask = org_model.view_rendering.get_virtual_image(
            #             src_color,
            #             src_mask,
            #             ref_depth,
            #             ref_invK,
            #             ref_K,
            #             T,
            #             source_scale
            #         )
            #         if frame_id == -1:
            #             org_warped_img = warped_img.clone()
            #             org_warped_mask = warped_mask.clone()

            #         if org_model.view_rendering.intensity_align:
            #             warped_img = org_model.view_rendering.get_norm_image_single(
            #                 ref_color,
            #                 ref_mask,
            #                 warped_img,
            #                 warped_mask
            #             )

            #         target_view[('color', frame_id, scale)] = warped_img
            #         target_view[('color_mask', frame_id, scale)] = warped_mask

            #     # spatio-temporal learning
            #     if org_model.view_rendering.spatio or org_model.view_rendering.spatio_temporal:
            #         for frame_id in org_model.view_rendering.frame_ids:
            #             overlap_img = torch.zeros_like(ref_color)
            #             overlap_mask = torch.zeros_like(ref_mask)

            #             for cur_index in org_model.view_rendering.rel_cam_list[cam]:
            #                 # for partial surround view training
            #                 if cur_index >= org_model.view_rendering.num_cams:
            #                     continue

            #                 src_color = inputs['color', frame_id, source_scale][:, cur_index, ...]
            #                 src_mask = inputs['mask'][:, cur_index, ...]
            #                 src_K = inputs[('K', source_scale)][:, cur_index, ...]

            #                 rel_pose = rel_pose_dict[(frame_id, cur_index)]
            #                 warped_img, warped_mask = org_model.view_rendering.get_virtual_image(
            #                     src_color,
            #                     src_mask,
            #                     ref_depth,
            #                     ref_invK,
            #                     src_K,
            #                     rel_pose,
            #                     source_scale
            #                 )

            #                 if org_model.view_rendering.intensity_align:
            #                     warped_img = org_model.view_rendering.get_norm_image_single(
            #                         ref_color,
            #                         ref_mask,
            #                         warped_img,
            #                         warped_mask
            #                     )

            #                 # assuming no overlap between warped images
            #                 overlap_img = overlap_img + warped_img
            #                 overlap_mask = overlap_mask + warped_mask

            #             target_view[('overlap', frame_id, scale)] = overlap_img
            #             target_view[('overlap_mask', frame_id, scale)] = overlap_mask
            # ########################################################################
            # cam_index = cam
            # org_cur_image = inputs['color', 0, 0]
            # org_prev_image = inputs['color', -1, 0]
            # org_next_image = inputs['color', 1, 0]
            # mask = inputs['mask']
            # intrinsic = inputs['K', 0]
            # inv_intrinsic = inputs['inv_K', 0]
            # extrinsic = inputs['extrinsics_inv']
            # inv_extrinsic = inputs['extrinsics']
            # distortion = None
            # cur_to_prev_pose = cur_to_prev_poses
            # cur_to_next_pose = cur_to_next_poses

            # # ref inputs
            # ref_color = org_cur_image[:, cam_index]  # inputs['color', 0, source_scale][:, cam, ...]
            # ref_mask = mask[:, cam_index]  # inputs['mask'][:, cam, ...]
            # ref_K = intrinsic[:, cam_index]  # inputs[('K', source_scale)][:, cam, ...]
            # ref_extrinsic = extrinsic[:, cam_index]
            # ref_distortion = distortion[:, cam_index] if distortion is not None else None

            # # target_view = outputs[('cam', cam)]
            # warped_views = {}

            # ref_depth = true_depth_map[:, cam_index]  # target_view[('depth', scale)]
            # for frame_id, T, src_color in zip(
            #     [-1, 1],
            #     [cur_to_prev_pose[:, cam_index], cur_to_next_pose[:, cam_index]],
            #     [org_prev_image[:, cam_index], org_next_image[:, cam_index]],
            # ):
            #     # for temporal learning
            #     src_mask = mask[:, cam_index]  # inputs['mask'][:, cam, ...]
            #     warped_img, warped_mask = view_renderer.get_virtual_image(
            #         src_img=src_color,
            #         src_mask=src_mask,
            #         dst_depth=ref_depth,
            #         src_intrinsic=ref_K,
            #         src_to_dst_transform=T,
            #         dst_intrinsic=ref_K,
            #         dst_extrinsic=ref_extrinsic,
            #         src_inv_intrinsic=inv_intrinsic[:, cam_index],
            #         src_inv_extrinsic=inv_extrinsic[:, cam_index],
            #         dst_inv_intrinsic=inv_intrinsic[:, cam_index],
            #         dst_inv_extrinsic=inv_extrinsic[:, cam_index],
            #         src_distortion=None,
            #         dst_distortion=ref_distortion,
            #     )
            #     if frame_id == -1:
            #         assert org_warped_img.shape == warped_img.shape
            #         assert torch.all(abs(org_warped_img - warped_img) < 1e-9), abs(org_warped_img - warped_img).max()
            #         assert org_warped_mask.shape == warped_mask.shape
            #         assert torch.all(abs(org_warped_mask - warped_mask) < 1e-9), abs(org_warped_mask - warped_mask).max()

            #     warped_img = view_renderer.get_norm_image_single(
            #         ref_color,
            #         ref_mask,
            #         warped_img,
            #         warped_mask
            #     )

            #     warped_views[('color', frame_id)] = warped_img
            #     warped_views[('color_mask', frame_id)] = warped_mask

            # # spatio-temporal learning
            # for frame_id, src_colors in zip(
            #     [0, -1, 1],
            #     [org_cur_image, org_prev_image, org_next_image],
            # ):
            #     overlap_img = torch.zeros_like(ref_color)
            #     overlap_mask = torch.zeros_like(ref_mask)

            #     for cur_index in org_model.view_rendering.rel_cam_list[cam]:
            #         # for partial surround view training
            #         src_color = src_colors[:, cur_index, ...]
            #         src_mask = mask[:, cur_index, ...]
            #         src_intrinsic = intrinsic[:, cur_index, ...]

            #         rel_pose = rel_pose_dict[(frame_id, cur_index)]
            #         warped_img, warped_mask = view_renderer.get_virtual_image(
            #             src_color,
            #             src_mask,
            #             ref_depth,
            #             src_intrinsic,
            #             rel_pose,
            #             ref_K,
            #             ref_extrinsic,
            #             ref_distortion,
            #         )

            #         warped_img = view_renderer.get_norm_image_single(
            #             ref_color,
            #             ref_mask,
            #             warped_img,
            #             warped_mask
            #         )

            #         # assuming no overlap between warped images
            #         overlap_img = overlap_img + warped_img
            #         overlap_mask = overlap_mask + warped_mask

            #     warped_views[('overlap', frame_id)] = overlap_img
            #     warped_views[('overlap_mask', frame_id)] = overlap_mask
            org_model.view_rendering(inputs, outputs, cam, org_rel_pose_dict)
            warped_views = view_renderer.forward(
                org_prev_image=inputs['color', -1, 0],
                org_cur_image=inputs['color', 0, 0],
                org_next_image=inputs['color', 1, 0],
                mask=inputs['mask'],
                intrinsic=inputs['K', 0],
                true_depth_map=true_depth_maps,
                cur_to_prev_pose=cur_to_prev_poses,
                cur_to_next_pose=cur_to_next_poses,
                cam_index=cam,
                neighbor_cam_indices=org_model.rel_cam_list[cam],
                rel_pose_dict=org_rel_pose_dict,
                extrinsic=inputs['extrinsics_inv'],
            )

            for k in warped_views:
                assert outputs['cam', cam][(*k, 0)].shape == warped_views[k].shape
                print(abs(outputs['cam', cam][(*k, 0)] - warped_views[k]).max())
                print(abs(outputs['cam', cam][(*k, 0)] - warped_views[k]).mean())
