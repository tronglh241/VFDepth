from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import conv1d, conv2d, pack_cam_feat
from .camera import PinHole


class VFNet(nn.Module):
    def __init__(
        self,
        feat_in_dim: int,
        feat_out_dim: int,
        fusion_level: int = 2,  # zero-based level, Resnet has 5-layer, e.g, 2 means 3rd layer.
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

        # define the 3D voxel space(follows the DDAD extrinsic coordinate -- x: forward, y: left, z: up)
        # define voxel end range in accordance with voxel_start_point, voxel_size, voxel_unit_size
        voxel_end_point = [voxel_start_point[i] + voxel_unit_size[i] * (voxel_size[i] - 1) for i in range(3)]
        self.voxel_unit_size = voxel_unit_size
        self.voxel_start_point = voxel_start_point
        self.voxel_end_point = voxel_end_point
        self.voxel_size = voxel_size
        self.x_dim, self.y_dim, self.z_dim = voxel_size

        # define a voxel space, [3, z, y, x], each voxel contains its 3D position
        voxel_grid = self.create_voxel_grid(voxel_start_point, voxel_end_point, voxel_size)
        self.voxel_points = voxel_grid.reshape(3, -1).T.contiguous()  # (x_dim * y_dim * z_dim, 3)

        # define grids in pixel space
        self.feat_height_2d = input_height // (2 ** (fusion_level + 1))  # 2D feature map height
        self.feat_width_2d = input_width // (2 ** (fusion_level + 1))  # 2D feature map width

        # define a depth grid for projection
        depth_bins = torch.linspace(proj_depth_start, proj_depth_end, proj_depth_bins)
        self.depth_grid = self.create_depth_grid(self.feat_width_2d, self.feat_height_2d, depth_bins)

    def create_voxel_grid(
        self,
        start_point: List[float],
        end_point: List[float],
        voxel_size: List[int],
    ):
        """
        output: [3, z_dim, y_dim, x_dim]
        [:, z, y, x] contains (x,y,z) 3D point
        """
        grids = [torch.linspace(start_point[i], end_point[i], voxel_size[i]) for i in range(3)]

        x_dim, y_dim, z_dim = voxel_size
        grids[0] = grids[0].view(1, 1, 1, x_dim)
        grids[1] = grids[1].view(1, 1, y_dim, 1)
        grids[2] = grids[2].view(1, z_dim, 1, 1)

        grids = [grid.expand(1, z_dim, y_dim, x_dim) for grid in grids]
        return torch.cat(grids, 0)

    def create_depth_grid(
        self,
        width: int,
        height: int,
        depth_bins: torch.Tensor,
    ):
        """
        output: [3, num_depths, height * width]
        output: [height, width, num_depths]
        """
        depth_layers = []
        for d in depth_bins:
            depth_layer = torch.full((height, width), d)
            depth_layers.append(depth_layer)
        depth_layers = torch.stack(depth_layers, dim=-1)
        return depth_layers

    def forward(
        self,
        mask,
        intrinsic,
        extrinsic,
        feats_agg,
    ):
        # backproject each per-pixel feature into 3D space (or sample per-pixel features for each voxel)
        voxel_feat_list, voxel_mask_list = self.backproject_into_voxel(feats_agg, mask, intrinsic, extrinsic)
        return voxel_feat_list, voxel_mask_list

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
            assert feats_img.shape[2:] == (self.feat_height_2d, self.feat_width_2d)

            batch_size, _, feat_height_2d, feat_width_2d = feats_img.size()

            mask_img = input_mask[:, cam, ...]
            mask_img = F.interpolate(mask_img, [feat_height_2d, feat_width_2d], mode='bilinear', align_corners=True)

            pin_hole = PinHole(
                width=feat_width_2d,
                height=feat_height_2d,
                extrinsic=extrinsic[:, cam],
                intrinsic=intrinsic[:, cam],
            )
            norm_points_2d, valid_points, points_depth = pin_hole.world_to_im(self.voxel_points.to(extrinsic.device))
            norm_points_2d = norm_points_2d.unsqueeze(-2)

            assert norm_points_2d.shape == (batch_size, self.voxel_points.shape[0], 1, 2)
            assert valid_points.shape == (batch_size, self.voxel_points.shape[0])
            assert points_depth.shape == (batch_size, self.voxel_points.shape[0])

            assert mask_img.shape[1] == 1

            roi_points = 0.5 < F.grid_sample(
                mask_img,
                norm_points_2d,
                mode='nearest',
                padding_mode='zeros',
                align_corners=True,
            ).squeeze(-1).squeeze(1)
            assert roi_points.shape == (batch_size, self.voxel_points.shape[0])

            valid_mask = valid_points * roi_points
            valid_mask = valid_mask.unsqueeze(1)

            # retrieve each per-pixel feature. [b, feat_dim, n_voxels, 1]
            feat_warped = F.grid_sample(
                feats_img,
                norm_points_2d,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            ).squeeze(-1)

            # concatenate relative depth as the feature. [b, feat_dim + 1, n_voxels]
            # TODO: this should be double-checked
            # points_depth.unsqueeze(1) should be divided by (self.voxel_size[0] * self.voxel_unit_size[0])
            feat_warped = torch.cat([feat_warped, points_depth.unsqueeze(1) / self.voxel_size[0]], dim=1)
            feat_warped = feat_warped * valid_mask

            voxel_feat_list.append(feat_warped)
            voxel_mask_list.append(valid_mask)

        return voxel_feat_list, voxel_mask_list


class PoseVFNet(VFNet):
    def __init__(
        self,
        feat_in_dim: int,
        feat_out_dim: int,
        fusion_level: int = 2,  # zero-based level, Resnet has 5-layer, e.g, 2 means 3rd layer.
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
        super(PoseVFNet, self).__init__(
            feat_in_dim=feat_in_dim,
            feat_out_dim=feat_out_dim,
            fusion_level=fusion_level,
            voxel_start_point=voxel_start_point,
            voxel_unit_size=voxel_unit_size,
            voxel_size=voxel_size,
            input_height=input_height,
            input_width=input_width,
            proj_depth_bins=proj_depth_bins,
            proj_depth_start=proj_depth_start,
            proj_depth_end=proj_depth_end,
            voxel_pre_dim=voxel_pre_dim,
        )

        encoder_dims = (feat_in_dim + 1) * self.z_dim
        stride = 2
        self.reduce_dim = nn.Sequential(
            *conv2d(encoder_dims, 256, kernel_size=3, stride=stride).children(),
            *conv2d(256, feat_out_dim, kernel_size=3, stride=stride).children(),
        )

    def forward(
        self,
        mask,
        intrinsic,
        extrinsic,
        feats_agg,
    ):
        voxel_feat_list, voxel_mask_list = super(PoseVFNet, self).forward(
            mask,
            intrinsic,
            extrinsic,
            feats_agg,
        )

        # compute overlap region
        voxel_mask_count = torch.sum(torch.cat(voxel_mask_list, dim=1), dim=1, keepdim=True)

        voxel_feat = torch.sum(torch.stack(voxel_feat_list, dim=1), dim=1, keepdim=False)
        voxel_feat = voxel_feat / (voxel_mask_count + 1e-7)

        b, c, _ = voxel_feat.shape
        voxel_feat = voxel_feat.view(b, c * self.z_dim, self.y_dim, self.x_dim)
        bev_feat = self.reduce_dim(voxel_feat)
        return bev_feat


class DepthVFNet(VFNet):
    def __init__(
        self,
        feat_in_dim: int,
        feat_out_dim: int,
        fusion_level: int = 2,  # zero-based level, Resnet has 5-layer, e.g, 2 means 3rd layer.
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
        super(DepthVFNet, self).__init__(
            feat_in_dim=feat_in_dim,
            feat_out_dim=feat_out_dim,
            fusion_level=fusion_level,
            voxel_start_point=voxel_start_point,
            voxel_unit_size=voxel_unit_size,
            voxel_size=voxel_size,
            input_height=input_height,
            input_width=input_width,
            proj_depth_bins=proj_depth_bins,
            proj_depth_start=proj_depth_start,
            proj_depth_end=proj_depth_end,
            voxel_pre_dim=voxel_pre_dim,
        )

        # voxel - preprocessing layer
        self.v_dim_o = [(feat_in_dim + 1) * 2, voxel_pre_dim]
        self.v_dim_no = [feat_in_dim + 1, voxel_pre_dim]

        self.conv_overlap = conv1d(self.v_dim_o[0], self.v_dim_o[1], kernel_size=1)
        self.conv_non_overlap = conv1d(self.v_dim_no[0], self.v_dim_no[1], kernel_size=1)

        encoder_dims = proj_depth_bins * self.v_dim_o[-1]
        stride = 1

        # channel dimension reduction
        self.reduce_dim = nn.Sequential(
            *conv2d(encoder_dims, 256, kernel_size=3, stride=stride).children(),
            *conv2d(256, feat_out_dim, kernel_size=3, stride=stride).children(),
        )

    def forward(
        self,
        mask,
        intrinsic,
        extrinsic,
        feats_agg,
    ):
        voxel_feat_list, voxel_mask_list = super(DepthVFNet, self).forward(
            mask,
            intrinsic,
            extrinsic,
            feats_agg,
        )

        # compute overlap region
        voxel_mask_count = torch.sum(torch.cat(voxel_mask_list, dim=1), dim=1, keepdim=True)

        # discriminatively process overlap and non_overlap regions using different MLPs
        voxel_non_overlap = self.preprocess_non_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
        voxel_overlap = self.preprocess_overlap(voxel_feat_list, voxel_mask_list, voxel_mask_count)
        voxel_feat = voxel_non_overlap + voxel_overlap

        # for each pixel, collect voxel features -> output image feature
        proj_feats = self.project_voxel_into_image(voxel_feat, intrinsic, extrinsic)
        proj_feat = pack_cam_feat(torch.stack(proj_feats, 1))
        return proj_feat

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
        assert len(voxel_feat_list) == len(voxel_mask_list)
        num_cams = len(voxel_feat_list)
        overlap_mask = (voxel_mask_count == 2)
        if num_cams == 3:
            feat1 = voxel_feat_list[0]
            feat2 = voxel_feat_list[1] + voxel_feat_list[2]
        elif num_cams == 6:
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
        intrinsic,
        extrinsic,
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
        for cam in range(intrinsic.shape[1]):
            # construct 3D point grid for each view
            pin_hole = PinHole(
                width=self.feat_width_2d,
                height=self.feat_height_2d,
                extrinsic=extrinsic[:, cam],
                intrinsic=intrinsic[:, cam],
            )

            cam_points = pin_hole.im_to_cam_map(self.depth_grid.to(intrinsic.device))
            batch_size, height, width, depth_bins, _ = cam_points.shape
            cam_points = cam_points.reshape(batch_size, height * width * depth_bins, 3)
            cam_points = pin_hole.cam_to_world(cam_points)
            cam_points = cam_points.view(batch_size, height * width, depth_bins, 3)
            cam_points = cam_points.permute(0, 2, 1, 3).reshape(batch_size, depth_bins * height * width, 3)

            # 3D grid_sample [b, n_voxels, 3], value: (x, y, z) point
            grid = cam_points

            for i in range(3):
                v_length = self.voxel_end_point[i] - self.voxel_start_point[i]
                grid[:, :, i] = (grid[:, :, i] - self.voxel_start_point[i]) / v_length * 2. - 1.

            grid = grid.view(b, depth_bins, self.feat_height_2d, self.feat_width_2d, 3)
            proj_feat = F.grid_sample(voxel_feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            proj_feat = proj_feat.view(b, depth_bins * self.v_dim_o[-1], self.feat_height_2d, self.feat_width_2d)

            # conv, reduce dimension
            proj_feat = self.reduce_dim(proj_feat)
            proj_feats.append(proj_feat)
        return proj_feats
