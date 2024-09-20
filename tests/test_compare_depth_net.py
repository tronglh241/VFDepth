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

    dataset = org_model.train_dataloader()
    inputs = next(iter(dataset))

    for key, ipt in inputs.items():
        if key not in _NO_DEVICE_KEYS:
            if 'context' in key:
                inputs[key] = [ipt[k].float().to(0) for k in range(len(inputs[key]))]
            else:
                inputs[key] = ipt.float().to(0)
    inputs['extrinsics_inv'] = torch.inverse(inputs['extrinsics'])

    # depth_feats = org_model.predict_depth(inputs)
    net = None
    if (org_model.mode != 'train') and org_model.ddp_enable:
        net = org_model.models['depth_net'].module
    else:
        net = org_model.models['depth_net']
    depth_feats = net(inputs)
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
    for cam in range(6):
        assert depth_feats['cam', cam]['disp', 0].shape == depth_maps[:, cam].shape
        print(abs(depth_feats['cam', cam]['disp', 0] - depth_maps[:, cam]).max())
        print(abs(depth_feats['cam', cam]['disp', 0] - depth_maps[:, cam]).mean())
        print(abs(depth_feats['cam', cam]['disp', 0] - depth_maps[:, cam]).std())
    # outputs = {}

    # # dictionary initialize
    # for cam in range(org_model.num_cams):
    #     outputs[('cam', cam)] = {}

    # lev = org_model.fusion_level

    # # packed images for surrounding view
    # sf_images = torch.stack([inputs[('color_aug', 0, 0)][:, cam, ...] for cam in range(org_model.num_cams)], 1)
    # packed_input = pack_cam_feat(sf_images)

    # # feature encoder
    # packed_feats = net.encoder(packed_input)
    # # aggregate feature H / 2^(lev+1) x W / 2^(lev+1)
    # _, _, up_h, up_w = packed_feats[lev].size()

    # packed_feats_list = packed_feats[lev:lev + 1] \
    #     + [F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True) for feat in packed_feats[lev + 1:]]

    # packed_feats_agg = net.conv1x1(torch.cat(packed_feats_list, dim=1))
    # feats_agg = unpack_cam_feat(packed_feats_agg, net.batch_size, net.num_cams)

    # # fusion_net, backproject each feature into the 3D voxel space
    # # fusion_dict = net.fusion_net(inputs, feats_agg)
    # mask = inputs['mask']
    # K = inputs['K', net.fusion_net.fusion_level + 1]
    # inv_K = inputs['inv_K', net.fusion_net.fusion_level + 1]
    # extrinsics = inputs['extrinsics']
    # extrinsics_inv = inputs['extrinsics_inv']

    # fusion_dict = {}
    # for cam in range(net.fusion_net.num_cams):
    #     fusion_dict[('cam', cam)] = {}

    # # device, dtype check, match dtype and device
    # sample_tensor = feats_agg[0, 0, ...]  # B, n_cam, c, h, w
    # net.fusion_net.type_check(sample_tensor)

    # backproject each per-pixel feature into 3D space (or sample per-pixel features for each voxel)
    # voxel_feat = net.fusion_net.backproject_into_voxel(feats_agg, mask, K, extrinsics_inv)

    ################################################
    # input_mask = mask
    # intrinsics = K
    # voxel_feat_list = []
    # voxel_mask_list = []

    # for cam in range(net.fusion_net.num_cams):
    #     feats_img = feats_agg[:, cam, ...]
    #     _, _, h_dim, w_dim = feats_img.size()

    #     mask_img = input_mask[:, cam, ...]
    #     mask_img = F.interpolate(mask_img, [h_dim, w_dim], mode='bilinear', align_corners=True)

    #     # 3D points in the voxel grid -> 3D points referenced at each view. [b, 3, n_voxels]
    #     ext_inv_mat = extrinsics_inv[:, cam, :3, :]
    #     v_pts_local = torch.matmul(ext_inv_mat, net.fusion_net.voxel_pts)

    #     # calculate pixel coordinate that each point are projected in the image. [b, n_voxels, 1, 2]
    #     K_mat = intrinsics[:, cam, :, :]
    #     pix_coords = net.fusion_net.calculate_sample_pixel_coords(K_mat, v_pts_local, w_dim, h_dim)

    #     # compute validity mask. [b, 1, n_voxels]
    #     valid_mask = net.fusion_net.calculate_valid_mask(mask_img, pix_coords, v_pts_local)

    #     # retrieve each per-pixel feature. [b, feat_dim, n_voxels, 1]
    #     feat_warped = F.grid_sample(feats_img, pix_coords, mode='bilinear', padding_mode='zeros', align_corners=True)
    #     # concatenate relative depth as the feature. [b, feat_dim + 1, n_voxels]
    #     feat_warped = torch.cat([
    #         feat_warped.squeeze(-1),
    #         v_pts_local[:, 2:3, :] / (net.fusion_net.voxel_size[0])
    #     ], dim=1)
    #     feat_warped = feat_warped * valid_mask.float()

    #     voxel_feat_list.append(feat_warped)
    #     voxel_mask_list.append(valid_mask)

    # # compute overlap region
    # voxel_mask_count = torch.sum(torch.cat(voxel_mask_list, dim=1), dim=1, keepdim=True)

    # # discriminatively process overlap and non_overlap regions using different MLPs
    # voxel_non_overlap = net.fusion_net.preprocess_non_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
    # voxel_overlap = net.fusion_net.preprocess_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
    # voxel_feat = voxel_non_overlap + voxel_overlap

    # # for each pixel, collect voxel features -> output image feature
    # proj_feats = net.fusion_net.project_voxel_into_image(voxel_feat, inv_K, extrinsics)
    # org_proj_feats = proj_feats
    # b, feat_dim, _ = voxel_feat.size()
    # voxel_feat = voxel_feat.view(b, feat_dim, net.fusion_net.z_dim, net.fusion_net.y_dim, net.fusion_net.x_dim)

    # proj_feats = []
    # for cam in range(net.fusion_net.num_cams):
    #     # construct 3D point grid for each view
    #     cam_points = torch.matmul(inv_K[:, cam, :3, :3], net.fusion_net.pixel_grid)
    #     cam_points = net.fusion_net.depth_grid * \
    #         cam_points.view(net.fusion_net.batch_size, 3, 1, net.fusion_net.num_pix)
    #     cam_points = torch.cat([cam_points, net.fusion_net.pixel_ones], dim=1)  # [b, 4, n_depthbins, n_pixels]
    #     cam_points = cam_points.view(net.fusion_net.batch_size, 4, -1)  # [b, 4, n_depthbins * n_pixels]
    #     # apply extrinsic: local 3D point -> global coordinate, [b, 3, n_depthbins * n_pixels]
    #     cam_points_ = cam_points.clone()
    #     points = torch.matmul(extrinsics[:, cam, :3, :], cam_points)
    #     org_cam_points = points.clone()

    #     # 3D grid_sample [b, n_voxels, 3], value: (x, y, z) point
    #     grid = points.permute(0, 2, 1)

    #     for i in range(3):
    #         v_length = net.fusion_net.voxel_end_p[i] - net.fusion_net.voxel_str_p[i]
    #         grid[:, :, i] = (grid[:, :, i] - net.fusion_net.voxel_str_p[i]) / v_length * 2. - 1.

    #     grid = grid.view(net.fusion_net.batch_size, net.fusion_net.proj_d_bins,
    #                      net.fusion_net.img_h, net.fusion_net.img_w, 3)
    #     proj_feat = F.grid_sample(voxel_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    #     proj_feat = proj_feat.view(
    #         b, net.fusion_net.proj_d_bins * net.fusion_net.v_dim_o[-1], net.fusion_net.img_h, net.fusion_net.img_w)
    #     # conv, reduce dimension
    #     proj_feat = net.fusion_net.reduce_dim(proj_feat)
    #     org_proj_feat = proj_feat.clone()
    #     proj_feats.append(proj_feat)
    #     #****************************************
    #     distortion = None
    #     intrinsic = inputs['K', 3]
    #     extrinsic = inputs['extrinsics_inv']
    #     inv_intrinsic = inputs['inv_K', 3]
    #     inv_extrinsic = inputs['extrinsics']

    #     # construct 3D point grid for each view
    #     if distortion is None:
    #         camera = PinHole(
    #             width=model.depth_net.fusion_net.feat_width_2d,
    #             height=model.depth_net.fusion_net.feat_height_2d,
    #             intrinsic=intrinsic[:, cam],
    #             extrinsic=extrinsic[:, cam],
    #             inv_intrinsic=inv_intrinsic[:, cam] if inv_intrinsic is not None else None,
    #             inv_extrinsic=inv_extrinsic[:, cam] if inv_extrinsic is not None else None,
    #         )
    #     else:
    #         camera = Fisheye(
    #             width=model.depth_net.fusion_net.feat_width_2d,
    #             height=model.depth_net.fusion_net.feat_height_2d,
    #             intrinsic=intrinsic[:, cam],
    #             extrinsic=extrinsic[:, cam],
    #             inv_intrinsic=inv_intrinsic[:, cam] if inv_intrinsic is not None else None,
    #             inv_extrinsic=inv_extrinsic[:, cam] if inv_extrinsic is not None else None,
    #             distortion=distortion[:, cam],
    #         )

    #     cam_points, _ = camera.im_to_cam_map(
    #         net.fusion_net.pixel_grid, model.depth_net.fusion_net.depth_grid.to(intrinsic.device))
    #     batch_size, height, width, depth_bins, _ = cam_points.shape
    #     batch_size, height, width, depth_bins, _ = cam_points.shape
    #     cam_points = cam_points.permute(0, 3, 1, 2, 4)
    #     cam_points = cam_points.reshape(batch_size, depth_bins * height * width, 3)
    #     cam_points_ = cam_points_.permute(0, 2, 1)[..., :3]
    #     assert cam_points_.shape == cam_points.shape, (cam_points_.shape, cam_points.shape)
    #     assert torch.all(abs(cam_points_ - cam_points) < 1e-9), abs(cam_points_ - cam_points).max()
    #     cam_points = camera.cam_to_world(cam_points)
    #     cam_points2 = cam_points.permute(0, 2, 1)
    #     assert org_cam_points.shape == cam_points2.shape, (org_cam_points.shape, cam_points2.shape)
    #     assert torch.all(abs(org_cam_points - cam_points2) < 1e-9), abs(org_cam_points - cam_points2).max()

    #     # 3D grid_sample [b, n_voxels, 3], value: (x, y, z) point
    #     grid = cam_points

    #     for i in range(3):
    #         v_length = model.depth_net.fusion_net.voxel_end_point[i] - model.depth_net.fusion_net.voxel_start_point[i]
    #         grid[:, :, i] = (grid[:, :, i] - model.depth_net.fusion_net.voxel_start_point[i]) / v_length * 2. - 1.

    #     grid = grid.view(b, depth_bins, model.depth_net.fusion_net.feat_height_2d,
    #                      model.depth_net.fusion_net.feat_width_2d, 3)
    #     proj_feat = F.grid_sample(voxel_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    #     proj_feat = proj_feat.view(
    #         b, depth_bins * model.depth_net.fusion_net.v_dim_o[-1], model.depth_net.fusion_net.feat_height_2d,
    #         model.depth_net.fusion_net.feat_width_2d)
    #     # conv, reduce dimension
    #     proj_feat = model.depth_net.fusion_net.reduce_dim(proj_feat)
    #     assert org_proj_feat.shape == proj_feat.shape
    #     assert torch.all(abs(org_proj_feat - proj_feat) < 1e-9), \
    #         abs(org_proj_feat - proj_feat).max()
    #     #****************************************
    # fusion_dict['proj_feat'] = pack_cam_feat(torch.stack(proj_feats, 1))

    # proj_feat = model.depth_net.fusion_net(
    #     mask=inputs['mask'],
    #     feats_agg=feats_agg,
    #     intrinsic=inputs['K', 3],
    #     extrinsic=inputs['extrinsics_inv'],
    #     inv_intrinsic=inputs['inv_K', 3],
    #     inv_extrinsic=inputs['extrinsics'],
    #     distortion=None,
    # )
    # assert fusion_dict['proj_feat'].shape == proj_feat.shape
    # print(abs(fusion_dict['proj_feat'] - proj_feat).max())
    # print(abs(fusion_dict['proj_feat'] - proj_feat).mean())

    # feat_in = packed_feats[:lev] + [fusion_dict['proj_feat']]
    # packed_depth_outputs = net.decoder(feat_in)

    # depth_outputs = unpack_cam_feat(packed_depth_outputs, net.batch_size, net.num_cams)

    # for cam in range(net.num_cams):
    #     for k in depth_outputs.keys():
    #         outputs[('cam', cam)][k] = depth_outputs[k][:, cam, ...]
