# Copyright (c) 2023 42dot. All rights reserved.
import torch

from .loss_util import compute_photometric_loss, compute_edg_smooth_loss, compute_masked_loss, compute_auto_masks
from flame.loss import LossBase

_EPSILON = 0.00001


class SingleCamLoss(LossBase):
    """
    Class for single camera(temporal only) loss calculation
    """

    def __init__(self, output_transform=lambda x: x):
        super(SingleCamLoss, self).__init__(output_transform)

    def compute_reproj_loss(
        self,
        cam_org_prev_image,
        cam_org_image,
        cam_org_next_image,
        cam_target_view,
        ref_mask,
    ):
        """
        This function computes reprojection loss using auto mask.
        """
        reprojection_losses = []
        for frame_id in [-1, 1]:
            reproj_loss_args = {
                'pred': cam_target_view[('color', frame_id)],
                'target': cam_org_image,
            }
            reprojection_losses.append(
                compute_photometric_loss(**reproj_loss_args)
            )

        reprojection_losses = torch.cat(reprojection_losses, 1)
        reprojection_loss, _ = torch.min(reprojection_losses, dim=1, keepdim=True)

        identity_reprojection_losses = []
        for cam_temporal_org_image in [cam_org_prev_image, cam_org_next_image]:
            identity_reproj_loss_args = {
                'pred': cam_temporal_org_image,
                'target': cam_org_image
            }
            identity_reprojection_losses.append(
                compute_photometric_loss(**identity_reproj_loss_args)
            )

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
        identity_reprojection_losses = identity_reprojection_losses \
            + _EPSILON * torch.randn(identity_reprojection_losses.shape, device=identity_reprojection_losses.device)
        identity_reprojection_loss, _ = torch.min(identity_reprojection_losses, dim=1, keepdim=True)

        # find minimum losses
        reprojection_auto_mask = compute_auto_masks(reprojection_loss, identity_reprojection_loss)
        reprojection_auto_mask *= ref_mask

        cam_target_view[('reproj_loss')] = reprojection_auto_mask * reprojection_loss
        cam_target_view[('reproj_mask')] = reprojection_auto_mask
        return compute_masked_loss(reprojection_loss, reprojection_auto_mask)

    def compute_smooth_loss(
        self,
        cam_org_image,
        cam_depth_map,
    ):
        """
        This function computes edge-aware smoothness loss for the disparity map.
        """
        mean_disp = cam_depth_map.mean(2, True).mean(3, True)
        norm_disp = cam_depth_map / (mean_disp + 1e-8)
        return compute_edg_smooth_loss(cam_org_image, norm_disp)

    def forward(
        self,
        cam_org_prev_image,
        cam_org_image,
        cam_org_next_image,
        cam_target_view,
        ref_mask,
        cam_depth_map,
    ):
        loss_dict = {}
        cam_loss = 0.  # loss across the multi-scale

        reprojection_loss = self.compute_reproj_loss(
            cam_org_prev_image,
            cam_org_image,
            cam_org_next_image,
            cam_target_view,
            ref_mask,
        )
        smooth_loss = self.compute_smooth_loss(
            cam_org_image,
            cam_depth_map,
        )

        cam_loss += reprojection_loss
        cam_loss += self.disparity_smoothness * smooth_loss

        loss_dict['reproj_loss'] = reprojection_loss.item()
        loss_dict['smooth'] = smooth_loss.item()

        return cam_loss, loss_dict
