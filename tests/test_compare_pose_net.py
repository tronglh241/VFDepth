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

    pose_pred = org_model.predict_pose(inputs)
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
    # axis_angle, translation = model.pose_net(
    #     cur_image=inputs['color_aug', -1, 0],
    #     next_image=inputs['color_aug', 0, 0],
    #     mask=inputs['mask'],
    #     intrinsic=inputs['K', org_model.models['pose_net'].fusion_level + 1],
    #     extrinsic=inputs['extrinsics_inv'],
    # )
    # cur_to_prev_poses = model.pose_net.compute_poses(
    #     axis_angle=axis_angle,
    #     translation=translation,
    #     invert=True,
    #     ref_extrinsic=inputs['extrinsics_inv'][:, 0:1],
    #     extrinsic=inputs['extrinsics_inv'],
    #     ref_inv_extrinsic=inputs['extrinsics'][:, 0:1],
    #     inv_extrinsic=inputs['extrinsics'],
    # )

    # # Current image to next image pose estimation
    # axis_angle, translation = model.pose_net(
    #     cur_image=inputs['color_aug', 0, 0],
    #     next_image=inputs['color_aug', 1, 0],
    #     mask=inputs['mask'],
    #     intrinsic=inputs['K', org_model.models['pose_net'].fusion_level + 1],
    #     extrinsic=inputs['extrinsics_inv'],
    # )
    # cur_to_next_poses = model.pose_net.compute_poses(
    #     axis_angle=axis_angle,
    #     translation=translation,
    #     invert=False,
    #     ref_extrinsic=inputs['extrinsics_inv'][:, 0:1],
    #     extrinsic=inputs['extrinsics_inv'],
    #     ref_inv_extrinsic=inputs['extrinsics'][:, 0:1],
    #     inv_extrinsic=inputs['extrinsics'],
    # )

    for cam in range(6):
        assert pose_pred['cam', cam]['cam_T_cam', 0, -1].shape == cur_to_prev_poses[:, cam].shape
        assert torch.all(abs(pose_pred['cam', cam]['cam_T_cam', 0, -1] - cur_to_prev_poses[:, cam]) < 1e-9)
    for cam in range(6):
        assert pose_pred['cam', cam]['cam_T_cam', 0, 1].shape == cur_to_next_poses[:, cam].shape
        assert torch.all(abs(pose_pred['cam', cam]['cam_T_cam', 0, 1] - cur_to_next_poses[:, cam]) < 1e-9)

    ###############################################################
    # outputs = {}

    # # initialize dictionary
    # for cam in range(org_pose_net.num_cams):
    #     outputs[('cam', cam)] = {}

    # lev = org_pose_net.fusion_level

    # # packed images for surrounding view
    # cur_image = inputs[('color_aug', org_pose_net.frame_ids[0], 0)]
    # next_image = inputs[('color_aug', org_pose_net.frame_ids[1], 0)]

    # pose_images = torch.cat([cur_image, next_image], 2)
    # packed_pose_images = pack_cam_feat(pose_images)

    # packed_feats = org_pose_net.encoder(packed_pose_images)

    # # aggregate feature H / 2^(lev+1) x W / 2^(lev+1)
    # _, _, up_h, up_w = packed_feats[lev].size()

    # packed_feats_list = packed_feats[lev:lev + 1] \
    #     + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats[lev + 1:]]

    # packed_feats_agg = org_pose_net.conv1x1(torch.cat(packed_feats_list, dim=1))
    # feats_agg = unpack_cam_feat(packed_feats_agg, org_pose_net.batch_size, org_pose_net.num_cams)
    # org_feats_agg = feats_agg
    ###############################################################

    # batch_size, num_cams, _, _, _ = cur_image.shape

    # # images (batch_size x num_cams x (channels x 2) x height x width)
    # images = torch.cat([cur_image, next_image], 2)
    # # packed_pose_images ((batch_size x num_cams) x (channels x 2) x height x width)
    # packed_pose_images = pack_cam_feat(images)

    # packed_feats = pose_net.encoder(packed_pose_images)

    # # aggregate feature H / 2^(lev + 1) x W / 2^(lev + 1)
    # _, _, up_h, up_w = packed_feats[pose_net.fusion_level].size()

    # packed_feats_list = (
    #     packed_feats[pose_net.fusion_level:pose_net.fusion_level + 1] + [
    #         F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True)
    #         for feat in packed_feats[pose_net.fusion_level + 1:]
    #     ]
    # )

    # # packed_feats_agg ((batch_size x num_cams) x fusion_feat_in_dim x feat_height x feat_width)
    # packed_feats_agg = pose_net.conv1x1(torch.cat(packed_feats_list, dim=1))
    # # feats_agg (batch_size x num_cams x fusion_feat_in_dim x feat_height x feat_width)
    # feats_agg = unpack_cam_feat(packed_feats_agg, batch_size, num_cams)
    ###############################################################

    # assert org_feats_agg.shape == feats_agg.shape
    # assert torch.all(abs(org_feats_agg - feats_agg) < 1e-9), \
    #     (abs(org_feats_agg - feats_agg).min(), abs(org_feats_agg - feats_agg).max())
    ###############################################################

    # org_vfnet = org_pose_net.fusion_net
    # vfnet = pose_net.fusion_net

    ###############################################################
    # mask = inputs['mask']
    # intrinsics = inputs['K', org_vfnet.fusion_level + 1]
    # inv_extrinsics = inputs['extrinsics']
    # extrinsics = inputs['extrinsics_inv']
    # org_bev_feat = org_vfnet(inputs, feats_agg)
    # bev_feat = vfnet(
    #     mask=mask,
    #     intrinsic=intrinsics,
    #     extrinsic=extrinsics,
    #     feats_agg=feats_agg,
    # )
    # assert org_bev_feat.shape == bev_feat.shape
    # assert torch.all(abs(org_bev_feat - bev_feat) < 1e-9)
    ###############################################################

    # net = None
    # if (org_model.mode != 'train') and org_model.ddp_enable:
    #     net = org_model.models['pose_net'].module
    # else:
    #     net = org_model.models['pose_net']
    # org_pose = org_model.pose.get_single_pose(net, inputs, None)

    # axis_angle, translation = model.pose_net(
    #     cur_image=inputs['color_aug', -1, 0],
    #     next_image=inputs['color_aug', 0, 0],
    #     mask=inputs['mask'],
    #     intrinsic=inputs['K', org_vfnet.fusion_level + 1],
    #     extrinsic=inputs['extrinsics_inv'],
    #     distortion=None,
    # )
    # pose = model.pose_net.compute_pose(axis_angle, translation, invert=True)
    # assert org_pose['cam_T_cam', 0, -1].shape == pose.shape
    # assert torch.all(abs(org_pose['cam_T_cam', 0, -1] - pose) < 1e-9)

    # axis_angle, translation = model.pose_net(
    #     cur_image=inputs['color_aug', 0, 0],
    #     next_image=inputs['color_aug', 1, 0],
    #     mask=inputs['mask'],
    #     intrinsic=inputs['K', org_vfnet.fusion_level + 1],
    #     extrinsic=inputs['extrinsics_inv'],
    #     distortion=None,
    # )
    # pose = model.pose_net.compute_pose(axis_angle, translation, invert=False)
    # assert org_pose['cam_T_cam', 0, 1].shape == pose.shape
    # assert torch.all(abs(org_pose['cam_T_cam', 0, 1] - pose) < 1e-9)

    # mask = inputs['mask']
    # K = inputs['K', org_vfnet.fusion_level + 1]
    # inv_K = inputs['inv_K', org_vfnet.fusion_level + 1]
    # extrinsics = inputs['extrinsics']
    # extrinsics_inv = inputs['extrinsics_inv']

    # fusion_dict = {}
    # for cam in range(org_vfnet.num_cams):
    #     fusion_dict[('cam', cam)] = {}

    # # device, dtype check, match dtype and device
    # sample_tensor = feats_agg[0, 0, ...]  # B, n_cam, c, h, w
    # org_vfnet.type_check(sample_tensor)

    # backproject each per-pixel feature into 3D space (or sample per-pixel features for each voxel)
    # input_mask = mask
    # intrinsics = K

    # voxel_feat_list = []
    # voxel_mask_list = []

    # for cam in range(org_vfnet.num_cams):
    #     feats_img = feats_agg[:, cam, ...]
    #     _, _, h_dim, w_dim = feats_img.size()

    #     mask_img = input_mask[:, cam, ...]
    #     mask_img = F.interpolate(mask_img, [h_dim, w_dim], mode='bilinear', align_corners=True)
    #     org_mask_img = mask_img

    #     # 3D points in the voxel grid -> 3D points referenced at each view. [b, 3, n_voxels]
    #     ext_inv_mat = extrinsics_inv[:, cam, :3, :]
    #     v_pts_local = torch.matmul(ext_inv_mat, org_vfnet.voxel_pts)

    #     # calculate pixel coordinate that each point are projected in the image. [b, n_voxels, 1, 2]
    #     K_mat = intrinsics[:, cam, :, :]
    #     pix_coords = org_vfnet.calculate_sample_pixel_coords(K_mat, v_pts_local, w_dim, h_dim)
    #     # v_pts = v_pts_local
    #     # _K = K_mat
    #     # cam_points = torch.matmul(_K[:, :3, :3], v_pts)
    #     # org_cam_points = cam_points
    #     # pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + org_vfnet.eps)
    #     # org_pix_coords_1 = pix_coords.clone()
    #     # if not torch.all(torch.isfinite(pix_coords)):
    #     #     pix_coords = torch.clamp(pix_coords, min=-w_dim * 2, max=w_dim * 2)
    #     # org_pix_coords_2 = pix_coords.clone()
    #     # pix_coords = pix_coords.view(org_vfnet.batch_size, 2, org_vfnet.n_voxels, 1)
    #     # pix_coords = pix_coords.permute(0, 2, 3, 1)
    #     # pix_coords[:, :, :, 0] = pix_coords[:, :, :, 0] / (w_dim - 1)
    #     # pix_coords[:, :, :, 1] = pix_coords[:, :, :, 1] / (h_dim - 1)
    #     # pix_coords = (pix_coords - 0.5) * 2
    #     # org_pix_coords_3 = pix_coords
    #     org_pix_coords = pix_coords

    #     # compute validity mask. [b, 1, n_voxels]
    #     valid_mask = org_vfnet.calculate_valid_mask(mask_img, pix_coords, v_pts_local)
    #     org_valid_mask = valid_mask

    #     # retrieve each per-pixel feature. [b, feat_dim, n_voxels, 1]
    #     feat_warped = F.grid_sample(feats_img, pix_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
    #     # concatenate relative depth as the feature. [b, feat_dim + 1, n_voxels]
    #     feat_warped = torch.cat([feat_warped.squeeze(-1), v_pts_local[:, 2:3, :] / (org_vfnet.voxel_size[0])], dim=1)
    #     feat_warped = feat_warped * valid_mask.float()
    #     org_feat_warped = feat_warped

    #     voxel_feat_list.append(feat_warped)
    #     voxel_mask_list.append(valid_mask)

    # org_voxel_feat_list = voxel_feat_list
    # org_voxel_mask_list = voxel_mask_list
    # ###############################################################
    # distortion = None
    # intrinsic = K
    # extrinsic = inputs['extrinsics_inv']
    # # voxel_feat_list = []
    # # voxel_mask_list = []

    # for cam in range(feats_agg.shape[1]):
    #     feats_img = feats_agg[:, cam, ...]
    #     assert feats_img.shape[2:] == (vfnet.feat_height_2d, vfnet.feat_width_2d)

    #     batch_size, _, feat_height_2d, feat_width_2d = feats_img.size()

    #     mask_img = input_mask[:, cam, ...]
    #     mask_img = F.interpolate(mask_img, [feat_height_2d, feat_width_2d], mode='bilinear', align_corners=True)
    #     # assert torch.all(abs(org_mask_img - mask_img) < 1e-9), (abs(org_mask_img - mask_img).max())

    #     if distortion is None:
    #         camera = PinHole(
    #             width=feat_width_2d,
    #             height=feat_height_2d,
    #             extrinsic=extrinsic[:, cam],
    #             intrinsic=intrinsic[:, cam],
    #         )
    #     else:
    #         camera = Fisheye(
    #             width=feat_width_2d,
    #             height=feat_height_2d,
    #             extrinsic=extrinsic[:, cam],
    #             intrinsic=intrinsic[:, cam],
    #             distortion=distortion[:, cam],
    #         )
    #     norm_points_2d, valid_points, points_depth = camera.world_to_im(vfnet.voxel_points.to(extrinsic.device))
    #     # points_3d = vfnet.voxel_points.to(extrinsic.device)
    #     # normalize = True
    #     # assert points_3d.shape[-1] == 3

    #     # points_3d = torch.cat([points_3d, torch.ones((*points_3d.shape[:-1], 1), device=points_3d.device)], dim=-1)
    #     # points_3d = camera.extrinsic[..., :3, :] @ torch.transpose(points_3d, -2, -1)
    #     # assert torch.all(abs(points_3d - v_pts_local) < 1e-9)
    #     # points_3d = torch.transpose(points_3d, -2, -1)

    #     # points_depth = points_3d[..., 2]
    #     # is_point_front = points_depth > 0

    #     # points_2d = camera.intrinsic[..., :3, :3] @ torch.transpose(points_3d, -2, -1)
    #     # assert org_cam_points.shape == points_2d.shape
    #     # assert torch.all(abs(points_2d - org_cam_points) < 1e-9)

    #     # points_2d = torch.transpose(points_2d, -2, -1)
    #     # points_2d = points_2d[..., :2] / (points_2d[..., 2:3] + camera.eps)
    #     # points_2d = points_2d[..., :2]
    #     # assert org_pix_coords_1.shape == torch.transpose(points_2d, -2, -1).shape
    #     # assert torch.all(abs(org_pix_coords_1 - torch.transpose(points_2d, -2, -1)) < 1e-9), \
    #     #     abs(org_pix_coords_1 - torch.transpose(points_2d, -2, -1)).max()

    #     # is_point_in_image = torch.logical_and(
    #     #     torch.logical_and(points_2d[..., 0] <= camera.width - 1, points_2d[..., 0] >= 0),
    #     #     torch.logical_and(points_2d[..., 1] <= camera.height - 1, points_2d[..., 1] >= 0),
    #     # )

    #     # valid_points = torch.logical_and(is_point_front, is_point_in_image)

    #     # if not torch.all(torch.isfinite(points_2d)):
    #     #     points_2d = torch.clamp(points_2d, min=-w_dim * 2, max=w_dim * 2)

    #     # if normalize:
    #     #     # points_2d = points_2d / torch.tensor([camera.width - 1, camera.height - 1], device=points_2d.device)
    #     #     points_2d[..., 0] = points_2d[..., 0] / (w_dim - 1)
    #     #     points_2d[..., 1] = points_2d[..., 1] / (h_dim - 1)
    #     #     points_2d = (points_2d - 0.5) * 2
    #     #     assert points_2d.unsqueeze(-2).shape == org_pix_coords_3.shape
    #     #     assert torch.all(abs(points_2d.unsqueeze(-2) - org_pix_coords_3) < 1e-9)
    #     # norm_points_2d = points_2d
    #     norm_points_2d = norm_points_2d.unsqueeze(-2)

    #     # assert org_pix_coords.shape == norm_points_2d.shape
    #     # assert torch.all(abs(org_pix_coords - norm_points_2d) < 1e-9)

    #     # assert org_pix_coords.shape == norm_points_2d.shape
    #     # assert torch.all(abs(org_pix_coords - norm_points_2d) < 1e-9), (abs(org_pix_coords - norm_points_2d).max())

    #     assert norm_points_2d.shape == (batch_size, vfnet.voxel_points.shape[0], 1, 2)
    #     assert valid_points.shape == (batch_size, vfnet.voxel_points.shape[0])
    #     assert points_depth.shape == (batch_size, vfnet.voxel_points.shape[0])

    #     assert mask_img.shape[1] == 1

    #     roi_points = 0.5 < F.grid_sample(
    #         mask_img,
    #         norm_points_2d,
    #         mode='nearest',
    #         padding_mode='zeros',
    #         align_corners=True,
    #     ).squeeze(-1).squeeze(1)
    #     assert roi_points.shape == (batch_size, vfnet.voxel_points.shape[0])

    #     valid_mask = valid_points * roi_points
    #     valid_mask = valid_mask.unsqueeze(1)

    #     # assert org_valid_mask.shape == valid_mask.shape
    #     # assert torch.all(abs(org_valid_mask.float() - valid_mask.float()) < 1e-9)

    #     # retrieve each per-pixel feature. [b, feat_dim, n_voxels, 1]
    #     feat_warped = F.grid_sample(
    #         feats_img,
    #         norm_points_2d,
    #         mode='bilinear',
    #         padding_mode='zeros',
    #         align_corners=True,
    #     ).squeeze(-1)

    #     # concatenate relative depth as the feature. [b, feat_dim + 1, n_voxels]
    #     # TODO: this should be double-checked
    #     # points_depth.unsqueeze(1) should be divided by (self.voxel_size[0] * self.voxel_unit_size[0])
    #     feat_warped = torch.cat([feat_warped, points_depth.unsqueeze(1) / vfnet.voxel_size[0]], dim=1)
    #     feat_warped = feat_warped * valid_mask

    #     voxel_feat_list.append(feat_warped)
    #     voxel_mask_list.append(valid_mask)
    #     # assert org_feat_warped.shape == feat_warped.shape
    #     # assert torch.all(abs(org_feat_warped - feat_warped) < 1e-9)

    # for i in range(len(org_voxel_feat_list)):
    #     assert torch.all(abs(voxel_feat_list[i] - org_voxel_feat_list[i]) < 1e-9)
    #     assert torch.all(abs(voxel_mask_list[i].float() - org_voxel_mask_list[i].float()) < 1e-9)
