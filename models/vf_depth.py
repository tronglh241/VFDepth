import torch
import torch.nn.functional as F
from torch import nn

from .fused_depth_net import FusedDepthNet
from .fused_pose_net import FusedPoseNet


class VFDepth(nn.Module):
    def __init__(
        self,
        fusion_level: int = 2,
        input_width: int = 640,
        input_height: int = 352,
        linear_depth: bool = False,
    ):
        super(VFDepth, self).__init__()
        self.pose_net = FusedPoseNet(
            input_width=input_width,
            input_height=input_height,
            fusion_level=fusion_level,
        )
        self.depth_net = FusedDepthNet(
            input_width=input_width,
            input_height=input_height,
            fusion_level=fusion_level,
        )
        self.fusion_level = fusion_level
        self.linear_depth = linear_depth

    def forward(
        self,
        prev_image,
        cur_image,
        next_image,
        mask,
        intrinsic,
        extrinsic,
        ref_extrinsic,
        distortion=None,
        fov=None,
        ref_inv_extrinsic=None,
        inv_intrinsic=None,
        inv_extrinsic=None,
    ):
        # prev_image (batch_size, num_cams, 3, height, width)
        # cur_image (batch_size, num_cams, 3, height, width)
        # next_image (batch_size, num_cams, 3, height, width)
        # mask (batch_size, num_cams, 1, height, width)
        # intrinsic (batch_size, num_cams, 4, 4)
        # extrinsic (batch_size, num_cams, 4, 4)
        # ref_extrinsic (batch_size, 1, 4, 4)

        intrinsic = intrinsic.clone()
        intrinsic[:, :, :2] /= (2 ** (self.fusion_level + 1))

        # Previous image to current image pose estimation
        axis_angle, translation = self.pose_net(
            cur_image=prev_image,
            next_image=cur_image,
            mask=mask,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            distortion=distortion,
            fov=fov,
        )
        cur_to_prev_poses = self.pose_net.compute_poses(
            axis_angle=axis_angle,
            translation=translation,
            invert=True,
            ref_extrinsic=ref_extrinsic,
            extrinsic=extrinsic,
            ref_inv_extrinsic=ref_inv_extrinsic,
            inv_extrinsic=inv_extrinsic,
        )

        # Current image to next image pose estimation
        axis_angle, translation = self.pose_net(
            cur_image=cur_image,
            next_image=next_image,
            mask=mask,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            distortion=distortion,
            fov=fov,
        )
        cur_to_next_poses = self.pose_net.compute_poses(
            axis_angle=axis_angle,
            translation=translation,
            invert=False,
            ref_extrinsic=ref_extrinsic,
            extrinsic=extrinsic,
            ref_inv_extrinsic=ref_inv_extrinsic,
            inv_extrinsic=inv_extrinsic,
        )

        # Depth estimation
        depth_maps = self.depth_net(
            images=cur_image,
            mask=mask,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            inv_intrinsic=inv_intrinsic,
            inv_extrinsic=inv_extrinsic,
            distortion=distortion,
            fov=fov,
        )

        # cur_to_prev_poses (batch_size, num_cams, 4, 4)
        # cur_to_next_poses (batch_size, num_cams, 4, 4)
        # depth_maps (batch_size, num_cams, 1, height, width)
        return cur_to_prev_poses, cur_to_next_poses, depth_maps

    def compute_relative_poses(
        self,
        cam_cur_to_prev_pose,
        cam_cur_to_next_pose,
        cam_inv_extrinsic,
        extrinsic,
        neighbor_cam_indices,
    ):
        """
        This function computes spatio & spatio-temporal transformation for images from different viewpoints.
        """
        rel_pose_dict = {}

        # current time step (spatio)
        for cur_index in neighbor_cam_indices:
            cur_extrinsic = extrinsic[:, cur_index, ...]
            rel_pose_dict[(0, cur_index)] = torch.matmul(cur_extrinsic, cam_inv_extrinsic)

        # different time step (spatio-temporal)
        for cur_index in neighbor_cam_indices:
            # for partial surround view training
            # assuming that extrinsic doesn't change
            rel_ext = rel_pose_dict[(0, cur_index)]
            rel_pose_dict[(-1, cur_index)] = torch.matmul(rel_ext, cam_cur_to_prev_pose)
            rel_pose_dict[(1, cur_index)] = torch.matmul(rel_ext, cam_cur_to_next_pose)
        return rel_pose_dict

    def compute_true_depth_maps(
        self,
        depth_maps,
        intrinsic,
        max_depth,
        min_depth,
        focal_length_scale,
        dst_width=None,
        dst_height=None,
    ):
        batch_size, num_cams, channels, height, width = depth_maps.shape
        dst_width = dst_width or width
        dst_height = dst_height or height

        depth_maps = depth_maps.view(batch_size * num_cams, channels, height, width)
        depth_maps = F.interpolate(depth_maps, [dst_height, dst_width], mode='bilinear', align_corners=False)
        depth_maps = depth_maps.view(batch_size, num_cams, channels, dst_height, dst_width)

        if not self.linear_depth:
            min_disp = 1 / max_depth
            max_disp = 1 / min_depth
            disp_range = max_disp - min_disp
            disp = min_disp + disp_range * depth_maps
            depth = 1 / disp
        else:
            depth = (1 - depth_maps) * max_depth
        return depth * intrinsic[:, :, 0:1, 0:1].unsqueeze(3) / focal_length_scale
