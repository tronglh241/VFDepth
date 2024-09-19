# Copyright (c) 2023 42dot. All rights reserved.
import torch
from pytorch3d.transforms import matrix_to_euler_angles

from .loss_util import compute_masked_loss, compute_photometric_loss
from .single_cam_loss import SingleCamLoss


class MultiCamLoss(SingleCamLoss):
    """
    Class for multi-camera(spatio & temporal) loss calculation
    """

    def __init__(
        self,
        disparity_smoothness,
        spatio_coeff,
        spatio_tempo_coeff,
    ):
        self.disparity_smoothness = disparity_smoothness
        self.spatio_coeff = spatio_coeff
        self.spatio_tempo_coeff = spatio_tempo_coeff

    def compute_spatio_loss(
        self,
        cam_org_image,
        cam_target_view,
        ref_mask,
    ):
        """
        This function computes spatial loss.
        """
        # self occlusion mask * overlap region mask
        spatio_mask = ref_mask * cam_target_view[('overlap_mask', 0)]
        loss_args = {
            'pred': cam_target_view[('overlap', 0)],
            'target': cam_org_image,
        }
        spatio_loss = compute_photometric_loss(**loss_args)

        cam_target_view[('overlap_mask', 0)] = spatio_mask
        return compute_masked_loss(spatio_loss, spatio_mask)

    def compute_spatio_tempo_loss(
        self,
        cam_org_image,
        cam_target_view,
        ref_mask,
        reproj_loss_mask,
    ):
        """
        This function computes spatio-temporal loss.
        """
        spatio_tempo_losses = []
        spatio_tempo_masks = []
        for frame_id in [-1, 1]:
            pred_mask = ref_mask * cam_target_view[('overlap_mask', frame_id)]
            pred_mask = pred_mask * reproj_loss_mask

            loss_args = {
                'pred': cam_target_view[('overlap', frame_id)],
                'target': cam_org_image,
            }

            spatio_tempo_losses.append(compute_photometric_loss(**loss_args))
            spatio_tempo_masks.append(pred_mask)

        # concatenate losses and masks
        spatio_tempo_losses = torch.cat(spatio_tempo_losses, 1)
        spatio_tempo_masks = torch.cat(spatio_tempo_masks, 1)

        # for the loss, take minimum value between reprojection loss and identity loss(moving object)
        # for the mask, take maximum value between reprojection mask and overlap mask to apply losses
        # on all the True values of masks.
        spatio_tempo_loss, _ = torch.min(spatio_tempo_losses, dim=1, keepdim=True)
        spatio_tempo_mask, _ = torch.max(spatio_tempo_masks.float(), dim=1, keepdim=True)

        return compute_masked_loss(spatio_tempo_loss, spatio_tempo_mask)

    def compute_pose_con_loss(
        self,
        ref_prev_to_cur_pose,
        ref_next_to_cur_pose,
        cam_prev_to_cur_pose,
        cam_next_to_cur_pose,
        ref_extrinsic,
        ref_inv_extrinsic,
        cam_extrinsic,
        cam_inv_extrinsic,
    ):
        """
        This function computes pose consistency loss in "Full surround monodepth from multiple cameras"
        """
        # ref_output = outputs[('cam', 0)]
        ref_ext = ref_inv_extrinsic
        ref_ext_inv = ref_inv_extrinsic

        # cur_output = outputs[('cam', cam)]
        cur_ext = cam_inv_extrinsic
        cur_ext_inv = cam_extrinsic

        trans_loss = 0.
        angle_loss = 0.

        for ref_T, cur_T in zip([
            [
                ref_prev_to_cur_pose,
                ref_next_to_cur_pose,
            ],
            [
                cam_prev_to_cur_pose,
                cam_next_to_cur_pose,
            ]
        ]):
            cur_T_aligned = ref_ext_inv @ cur_ext @ cur_T @ cur_ext_inv @ ref_ext

            ref_ang = matrix_to_euler_angles(ref_T[:, :3, :3], 'XYZ')
            cur_ang = matrix_to_euler_angles(cur_T_aligned[:, :3, :3], 'XYZ')

            ang_diff = torch.norm(ref_ang - cur_ang, p=2, dim=1).mean()
            t_diff = torch.norm(ref_T[:, :3, 3] - cur_T_aligned[:, :3, 3], p=2, dim=1).mean()

            trans_loss += t_diff
            angle_loss += ang_diff

        pose_loss = (trans_loss + 10 * angle_loss) / 2
        return pose_loss

    def __call__(
        self,
        cam_org_prev_image,
        cam_org_image,
        cam_org_next_image,
        cam_target_view,
        cam_depth_map,
        cam_mask,
    ):
        loss_dict = {}
        cam_loss = 0.  # loss across the multi-scale

        reprojection_loss = self.compute_reproj_loss(
            cam_org_prev_image=cam_org_prev_image,
            cam_org_image=cam_org_image,
            cam_org_next_image=cam_org_next_image,
            cam_target_view=cam_target_view,
            ref_mask=cam_mask,
        )
        smooth_loss = self.compute_smooth_loss(
            cam_org_image=cam_org_image,
            cam_depth_map=cam_depth_map,
        )
        spatio_loss = self.compute_spatio_loss(
            cam_org_image=cam_org_image,
            cam_target_view=cam_target_view,
            ref_mask=cam_mask,
        )

        spatio_tempo_loss = self.compute_spatio_tempo_loss(
            cam_org_image=cam_org_image,
            cam_target_view=cam_target_view,
            ref_mask=cam_mask,
            reproj_loss_mask=cam_target_view[('reproj_mask')],
        )

        cam_loss += reprojection_loss
        cam_loss += self.disparity_smoothness * smooth_loss
        cam_loss += self.spatio_coeff * spatio_loss + self.spatio_tempo_coeff * spatio_tempo_loss

        ##########################
        # for logger
        ##########################
        loss_dict['reproj_loss'] = reprojection_loss.item()
        loss_dict['spatio_loss'] = spatio_loss.item()
        loss_dict['spatio_tempo_loss'] = spatio_tempo_loss.item()
        loss_dict['smooth'] = smooth_loss.item()

        return cam_loss, loss_dict
