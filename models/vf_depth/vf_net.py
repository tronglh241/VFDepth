from typing import List

import torch
import torch.nn.functional as F
from network.blocks import conv1d, conv2d, pack_cam_feat
from torch import nn


class VFNet(nn.Module):
    def __init__(
        self,
        feat_in_dim: int,
        feat_out_dim: int,
        model_type: str = 'depth',
        voxel_start_point: List[float] = [-50.0, -50.0, -15.0],  # voxel start point [x, y, z]
        voxel_unit_size: List[float] = [1.0, 1.0, 1.5],  # size of unit voxel in (m) [x, y, z]
        voxel_size: List[int] = [100, 100, 20],  # num of voxels in each dimension [x, y, z]
        input_height: int = 352,
        input_width: int = 640,
        proj_depth_bins: int = 50,
        proj_depth_start: int = 2,
        proj_depth_end: int = 50,
        voxel_pre_dim: int = 64,
    ):
        super(VFNet, self).__init__()
        self.eps = 1e-8
        self.model_type = model_type

        # define the 3D voxel space(follows the DDAD extrinsic coordinate -- x: forward, y: left, z: up)
        # define voxel end range in accordance with voxel_start_point, voxel_size, voxel_unit_size
        voxel_end_point = [voxel_start_point[i] + voxel_unit_size[i] * (voxel_size[i] - 1) for i in range(3)]
        self.voxel_unit_size = voxel_unit_size
        self.voxel_start_point = voxel_start_point
        self.voxel_end_point = voxel_end_point
        self.voxel_size = voxel_size

        # define a voxel space, [1, 3, z, y, x], each voxel contains its 3D position
        voxel_grid = self.create_voxel_grid(voxel_start_point, voxel_end_point, voxel_size)
        b, _, self.z_dim, self.y_dim, self.x_dim = voxel_grid.size()
        self.n_voxels = self.z_dim * self.y_dim * self.x_dim
        self.voxel_pts = torch.cat(
            [
                voxel_grid.view(b, 3, self.n_voxels),
                torch.ones(b, 1, self.n_voxels),
            ],
            dim=1,
        )

        # define grids in pixel space
        self.proj_depth_bins = proj_depth_bins
        self.img_h = input_height // (2 ** (self.fusion_level + 1))  # 2D feature map height
        self.img_w = input_width // (2 ** (self.fusion_level + 1))  # 2D feature map width
        self.num_pix = self.img_h * self.img_w
        self.pixel_grid = self.create_pixel_grid(self.img_h, self.img_w)
        self.pixel_ones = torch.ones(b, 1, proj_depth_bins, self.num_pix)  # this is for homogeneous transformation

        # define a depth grid for projection
        depth_bins = torch.linspace(proj_depth_start, proj_depth_end, proj_depth_bins)
        self.depth_grid = self.create_depth_grid(self.num_pix, proj_depth_bins, depth_bins)

        # depth fusion(process overlap and non-overlap regions)
        if model_type == 'depth':
            # voxel - preprocessing layer
            self.v_dim_o = [(feat_in_dim + 1) * 2, voxel_pre_dim]
            self.v_dim_no = [feat_in_dim + 1, voxel_pre_dim]

            self.conv_overlap = conv1d(self.v_dim_o[0], self.v_dim_o[1], kernel_size=1)
            self.conv_non_overlap = conv1d(self.v_dim_no[0], self.v_dim_no[1], kernel_size=1)

            encoder_dims = proj_depth_bins * self.v_dim_o[-1]
            stride = 1

        else:
            encoder_dims = (feat_in_dim + 1) * self.z_dim
            stride = 2

        # channel dimension reduction
        self.reduce_dim = nn.Sequential(
            *conv2d(encoder_dims, 256, kernel_size=3, stride=stride).children(),
            *conv2d(256, feat_out_dim, kernel_size=3, stride=stride).children(),
        )

    def create_voxel_grid(
        self,
        start_point: List[float],
        end_point: List[float],
        voxel_size: List[int],
    ):
        """
        output: [batch, 3, z_dim, y_dim, x_dim]
        [b, :, z, y, x] contains (x,y,z) 3D point
        """
        grids = [torch.linspace(start_point[i], end_point[i], voxel_size[i]) for i in range(3)]

        x_dim, y_dim, z_dim = voxel_size
        grids[0] = grids[0].view(1, 1, 1, 1, x_dim)
        grids[1] = grids[1].view(1, 1, 1, y_dim, 1)
        grids[2] = grids[2].view(1, 1, z_dim, 1, 1)

        # TODO: batch_size should be removed here
        batch_size = 2  # hard coded
        grids = [grid.expand(batch_size, 1, z_dim, y_dim, x_dim) for grid in grids]
        return torch.cat(grids, 1)

    def create_pixel_grid(
        self,
        height: int,
        width: int,
    ):
        """
        output: [batch, 3, height * width]
        """
        # TODO: batch_size should be removed
        batch_size = 2  # hard coded
        grid_xy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        pix_coords = torch.stack(grid_xy, axis=0).unsqueeze(0).view(1, 2, height * width)
        pix_coords = pix_coords.repeat(batch_size, 1, 1)
        pix_coords = torch.cat([pix_coords, torch.ones(batch_size, 1, height * width)], 1)
        return pix_coords

    def create_depth_grid(
        self,
        n_pixels: int,
        n_depth_bins: int,
        depth_bins: torch.Tensor,
    ):
        """
        output: [batch, 3, num_depths, height * width]
        """
        # TODO: batch_size should be removed
        batch_size = 2
        depth_layers = []
        for d in depth_bins:
            depth_layer = torch.ones((1, n_pixels)) * d
            depth_layers.append(depth_layer)
        depth_layers = torch.cat(depth_layers, dim=0).view(1, 1, n_depth_bins, n_pixels)
        depth_layers = depth_layers.expand(batch_size, 3, n_depth_bins, n_pixels)
        return depth_layers

    def forward(
        self,
        mask,
        intrinsic,
        inv_intrisic,
        extrinsic,
        inv_extrinsic,
        feats_agg,
    ):
        # TODO: extrinsic  and inv_extrinsic should be swapped, this perhaps only works with nuScene calibration

        # K = inputs['K', self.fusion_level + 1]
        # inv_K = inputs['inv_K', self.fusion_level + 1]
        # extrinsics = inputs['extrinsics']
        # extrinsics_inv = inputs['extrinsics_inv']

        fusion_dict = {}
        for cam in range(self.num_cams):
            fusion_dict[('cam', cam)] = {}

        # device, dtype check, match dtype and device
        # sample_tensor = feats_agg[0, 0, ...]  # B, n_cam, c, h, w
        # self.type_check(sample_tensor)

        # backproject each per-pixel feature into 3D space (or sample per-pixel features for each voxel)
        voxel_feat = self.backproject_into_voxel(feats_agg, mask, intrinsic, inv_extrinsic)

        if self.model_type == 'depth':
            # for each pixel, collect voxel features -> output image feature
            proj_feats = self.project_voxel_into_image(voxel_feat, inv_intrisic, extrinsic)
            fusion_dict['proj_feat'] = pack_cam_feat(torch.stack(proj_feats, 1))

            # # with view augmentation
            # if self.aug_depth:
            #     # extrinsics
            #     inputs['extrinsics_aug'] = self.augment_extrinsics(extrinsic)
            #     proj_feats = self.project_voxel_into_image(voxel_feat, inv_intrisic, inputs['extrinsics_aug'])
            #     fusion_dict['proj_feat_aug'] = pack_cam_feat(torch.stack(proj_feats, 1))

            # # synthesis visualization
            # if self.syn_visualize:
            #     def _get_proj_feat(inv_K, ang_x, ang_y, ang_z):
            #         angle_mat = axis_angle_to_matrix(torch.tensor([ang_x, ang_y, ang_z])[None, :])  # 3x3
            #         b, c, _, _ = extrinsics.size()
            #         tform_mat = torch.eye(4)[None, None]
            #         tform_mat[:, :, :3, :3] = angle_mat
            #         tform_mat = tform_mat.repeat(b, c, 1, 1).to(device=extrinsics.device, dtype=extrinsics.dtype)
            #         proj_feats = self.project_voxel_into_image(voxel_feat, inv_K, tform_mat @ extrinsics)
            #         return proj_feats[0]

            #     fusion_dict['syn_feat'] = []

            #     # augmented intrinsics and extrinsics
            #     aug_params = aug_depth_params(K)
            #     for param in aug_params:
            #         fusion_dict['syn_feat'] += [_get_proj_feat(*param)]
            return fusion_dict

        elif self.model_type == 'pose':
            b, c, _ = voxel_feat.shape
            voxel_feat = voxel_feat.view(b, c * self.z_dim,
                                         self.y_dim, self.x_dim)
            bev_feat = self.reduce_dim(voxel_feat)
            return bev_feat

        else:
            raise NotImplementedError

    def backproject_into_voxel(
        self,
        feats_agg,
        input_mask,
        intrinsic,
        extrinsic,
    ):
        """
        This function backprojects 2D features into 3D voxel coordinate using intrinsic and extrinsic of each camera.
        Self-occluded regions are removed by using the projected mask in 3D voxel coordinate.
        """
        voxel_feat_list = []
        voxel_mask_list = []

        for cam in range(feats_agg.shape[1]):
            feats_img = feats_agg[:, cam, ...]
            _, _, h_dim, w_dim = feats_img.size()

            mask_img = input_mask[:, cam, ...]
            mask_img = F.interpolate(mask_img, [h_dim, w_dim], mode='bilinear', align_corners=True)

            # 3D points in the voxel grid -> 3D points referenced at each view. [b, 3, n_voxels]
            cam_extrinsic = extrinsic[:, cam, :3, :]
            v_pts_local = torch.matmul(cam_extrinsic, self.voxel_pts)

            # calculate pixel coordinate that each point are projected in the image. [b, n_voxels, 1, 2]
            cam_intrinsic = intrinsic[:, cam, :, :]
            pix_coords = self.calculate_sample_pixel_coords(cam_intrinsic, v_pts_local, w_dim, h_dim)

            # compute validity mask. [b, 1, n_voxels]
            valid_mask = self.calculate_valid_mask(mask_img, pix_coords, v_pts_local)

            # retrieve each per-pixel feature. [b, feat_dim, n_voxels, 1]
            feat_warped = F.grid_sample(
                feats_img, pix_coords,
                mode='bilinear', padding_mode='zeros', align_corners=True,
            )
            # concatenate relative depth as the feature. [b, feat_dim + 1, n_voxels]
            # TODO: this should be double-checked
            # v_pts_local[:, 2:3, :] should be divided by (self.voxel_size[0] * self.voxel_unit_size[0])
            feat_warped = torch.cat([feat_warped.squeeze(-1), v_pts_local[:, 2:3, :] / self.voxel_size[0]], dim=1)
            feat_warped = feat_warped * valid_mask.float()

            voxel_feat_list.append(feat_warped)
            voxel_mask_list.append(valid_mask)

        # compute overlap region
        voxel_mask_count = torch.sum(torch.cat(voxel_mask_list, dim=1), dim=1, keepdim=True)

        if self.model_type == 'depth':
            # discriminatively process overlap and non_overlap regions using different MLPs
            voxel_non_overlap = self.preprocess_non_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
            voxel_overlap = self.preprocess_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
            voxel_feat = voxel_non_overlap + voxel_overlap

        elif self.model_type == 'pose':
            voxel_feat = torch.sum(torch.stack(voxel_feat_list, dim=1), dim=1, keepdim=False)
            voxel_feat = voxel_feat / (voxel_mask_count + 1e-7)

        else:
            raise NotImplementedError

        return voxel_feat

    def calculate_sample_pixel_coords(
        self,
        intrinsic,
        voxel_points,
        img_width,
        img_height,
    ):
        """
        This function calculates pixel coords for each point([batch, n_voxels, 1, 2]) to sample the per-pixel feature.
        """
        cam_points = torch.matmul(intrinsic[:, :3, :3], voxel_points)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)

        if not torch.all(torch.isfinite(pix_coords)):
            pix_coords = torch.clamp(pix_coords, min=-img_width * 2, max=img_width * 2)

        # TODO: batch_size should be removed
        batch_size = 2  # hard coded
        pix_coords = pix_coords.view(batch_size, 2, self.n_voxels, 1)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[:, :, :, 0] = pix_coords[:, :, :, 0] / (img_width - 1)
        pix_coords[:, :, :, 1] = pix_coords[:, :, :, 1] / (img_height - 1)
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

    def calculate_valid_mask(
        self,
        mask_img,
        pix_coords,
        v_pts_local,
    ):
        """
        This function creates valid mask in voxel coordinate by projecting self-occlusion mask to 3D voxel coords.
        """
        # compute validity mask, [b, 1, n_voxels, 1]
        mask_selfocc = (
            F.grid_sample(mask_img, pix_coords, mode='nearest', padding_mode='zeros', align_corners=True) > 0.5
        )
        # discard points behind the camera, [b, 1, n_voxels]
        mask_depth = (v_pts_local[:, 2:3, :] > 0)
        # compute validity mask, [b, 1, n_voxels, 1]
        pix_coords_mask = pix_coords.permute(0, 3, 1, 2)
        mask_oob = ~(torch.logical_or(pix_coords_mask > 1, pix_coords_mask < -1).sum(dim=1, keepdim=True) > 0)
        valid_mask = mask_selfocc.squeeze(-1) * mask_depth * mask_oob.squeeze(-1)
        return valid_mask

    def preprocess_non_overlap(
        self,
        voxel_feat_list,
        voxel_mask_list,
        voxel_mask_count,
    ):
        """
        This function applies 1x1 convolutions to features from non-overlapping features.
        """
        non_overlap_mask = (voxel_mask_count == 1)
        voxel = sum(voxel_feat_list)
        voxel = voxel * non_overlap_mask.float()

        for conv_no in self.conv_non_overlap:
            voxel = conv_no(voxel)
        return voxel * non_overlap_mask.float()

    def preprocess_overlap(
        self,
        voxel_feat_list,
        voxel_mask_list,
        voxel_mask_count,
    ):
        """
        This function applies 1x1 convolutions on overlapping features.
        Camera configuration [0,1,2] or [0,1,2,3,4,5]:
                        3 1
            rear cam <- 5   0 -> front cam
                        4 2
        """
        overlap_mask = (voxel_mask_count == 2)
        if self.num_cams == 3:
            feat1 = voxel_feat_list[0]
            feat2 = voxel_feat_list[1] + voxel_feat_list[2]
        elif self.num_cams == 6:
            feat1 = voxel_feat_list[0] + voxel_feat_list[3] + voxel_feat_list[4]
            feat2 = voxel_feat_list[1] + voxel_feat_list[2] + voxel_feat_list[5]
        else:
            raise NotImplementedError

        voxel = torch.cat([feat1, feat2], dim=1)
        for conv_o in self.conv_overlap:
            voxel = conv_o(voxel)
        return voxel * overlap_mask.float()

    def project_voxel_into_image(
        self,
        voxel_feat,
        inv_intrisic,
        inv_extrinsic,
    ):
        """
        This function projects voxels into 2D image coordinate.
        [b, feat_dim, n_voxels] -> [b, feat_dim, d, h, w]
        """
        # define depth bin
        # [b, feat_dim, n_voxels] -> [b, feat_dim, d, h, w]
        b, feat_dim, _ = voxel_feat.size()
        voxel_feat = voxel_feat.view(b, feat_dim, self.z_dim, self.y_dim, self.x_dim)

        proj_feats = []
        for cam in range(inv_intrisic.shape[1]):
            # construct 3D point grid for each view
            cam_points = torch.matmul(inv_intrisic[:, cam, :3, :3], self.pixel_grid)
            cam_points = self.depth_grid * cam_points.view(b, 3, 1, self.num_pix)
            cam_points = torch.cat([cam_points, self.pixel_ones], dim=1)  # [b, 4, n_depthbins, n_pixels]
            cam_points = cam_points.view(b, 4, -1)  # [b, 4, n_depthbins * n_pixels]

            # apply extrinsic: local 3D point -> global coordinate, [b, 3, n_depthbins * n_pixels]
            points = torch.matmul(inv_extrinsic[:, cam, :3, :], cam_points)

            # 3D grid_sample [b, n_voxels, 3], value: (x, y, z) point
            grid = points.permute(0, 2, 1)

            for i in range(3):
                v_length = self.voxel_end_point[i] - self.voxel_start_point[i]
                grid[:, :, i] = (grid[:, :, i] - self.voxel_start_point[i]) / v_length * 2. - 1.

            grid = grid.view(b, self.proj_depth_bins, self.img_h, self.img_w, 3)
            proj_feat = F.grid_sample(voxel_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            proj_feat = proj_feat.view(b, self.proj_depth_bins * self.v_dim_o[-1], self.img_h, self.img_w)

            # conv, reduce dimension
            proj_feat = self.reduce_dim(proj_feat)
            proj_feats.append(proj_feat)
        return proj_feats
