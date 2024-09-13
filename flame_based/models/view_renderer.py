import torch
import torch.nn.functional as F
from torch import nn

from .camera import PinHole


class ViewRenderer(nn.Module):
    def get_virtual_image(
        self,
        src_img,
        src_mask,
        dst_depth,
        src_intrinsic,
        src_to_dst_transform,
        dst_intrinsic,
        dst_extrinsic,
    ):
        """
        This function warps source image to target image using backprojection and reprojection process.
        """
        # do reconstruction for target from source
        # dst_depth (batch_size, 1, height, width)
        batch_size, _, height, width = dst_depth.shape

        dst_pin_hole = PinHole(
            width=width,
            height=height,
            extrinsic=dst_extrinsic,
            intrinsic=dst_intrinsic,
        )
        src_pin_hole = PinHole(
            width=width,
            height=height,
            extrinsic=src_to_dst_transform,
            intrinsic=src_intrinsic,
        )

        points_3d = dst_pin_hole.im_to_cam_map(dst_depth.permute(0, 2, 3, 1))
        # points_3d (batch_size, height, width, 1, 3)
        assert points_3d.shape[-2] == 1

        points_3d = points_3d.squeeze(-2)
        points_3d = points_3d.view(batch_size, height * width, 3)

        pix_coords, _, _ = src_pin_hole.world_to_im(points_3d)
        assert pix_coords.shape == (batch_size, width * height, 2)
        pix_coords = pix_coords.view(batch_size, height, width, 2)

        img_warped = F.grid_sample(src_img, pix_coords, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
        mask_warped = F.grid_sample(src_mask, pix_coords, mode='nearest',
                                    padding_mode='zeros', align_corners=True)

        # nan handling
        inf_img_regions = torch.isnan(img_warped)
        img_warped[inf_img_regions] = 2.0
        inf_mask_regions = torch.isnan(mask_warped)
        mask_warped[inf_mask_regions] = 0

        pix_coords = pix_coords.permute(0, 3, 1, 2)
        invalid_mask = torch.logical_or(pix_coords > 1,
                                        pix_coords < -1).sum(dim=1, keepdim=True) > 0
        return img_warped, (~invalid_mask).float() * mask_warped

    def get_mean_std(self, feature, mask):
        """
        This function returns mean and standard deviation of the overlapped features.
        """
        _, c, h, w = mask.size()
        mean = (feature * mask).sum(dim=(1, 2, 3), keepdim=True) / (mask.sum(dim=(1, 2, 3), keepdim=True) + 1e-8)
        var = ((feature - mean) ** 2).sum(dim=(1, 2, 3), keepdim=True) / (c * h * w)
        return mean, torch.sqrt(var + 1e-16)

    def get_norm_image_single(self, src_img, src_mask, warp_img, warp_mask):
        """
        obtain normalized warped images using the mean and the variance from the overlapped regions of the target frame.
        """
        warp_mask = warp_mask.detach()

        with torch.no_grad():
            mask = (src_mask * warp_mask).bool()
            if mask.size(1) != 3:
                mask = mask.repeat(1, 3, 1, 1)

            mask_sum = mask.sum(dim=(-3, -2, -1))
            # skip when there is no overlap
            if torch.any(mask_sum == 0):
                return warp_img

            s_mean, s_std = self.get_mean_std(src_img, mask)
            w_mean, w_std = self.get_mean_std(warp_img, mask)

        norm_warp = (warp_img - w_mean) / (w_std + 1e-8) * s_std + s_mean
        return norm_warp * warp_mask.float()

    def forward(
        self,
        org_prev_image,
        org_image,
        org_next_image,
        mask,
        intrinsic,
        inv_intrinsic,
        true_depth_map,
        prev_to_cur_pose,
        next_to_cur_pose,
        cam_index,
        neighbor_cam_indices,
        rel_pose_dict,
        extrinsic,
    ):
        # predict images for each scale(default = scale 0 only)
        # source_scale = 0

        # ref inputs
        ref_color = org_image[:, cam_index]  # inputs['color', 0, source_scale][:, cam, ...]
        ref_mask = mask[:, cam_index]  # inputs['mask'][:, cam, ...]
        ref_K = intrinsic[:, cam_index]  # inputs[('K', source_scale)][:, cam, ...]
        ref_extrinsic = extrinsic[:, cam_index]

        # target_view = outputs[('cam', cam)]
        warped_views = {}

        ref_depth = true_depth_map[:, cam_index]  # target_view[('depth', scale)]
        for frame_id, T, src_color in zip(
            [-1, 1],
            [prev_to_cur_pose[:, cam_index], next_to_cur_pose[:, cam_index]],
            [org_prev_image[:, cam_index], org_next_image[:, cam_index]],
        ):
            # for temporal learning
            src_mask = mask[:, cam_index]  # inputs['mask'][:, cam, ...]
            warped_img, warped_mask = self.get_virtual_image(
                src_color,
                src_mask,
                ref_depth,
                ref_K,
                T,
                ref_K,
                ref_extrinsic,
            )

            warped_img = self.get_norm_image_single(
                ref_color,
                ref_mask,
                warped_img,
                warped_mask
            )

            warped_views[('color', frame_id)] = warped_img
            warped_views[('color_mask', frame_id)] = warped_mask

        # spatio-temporal learning
        for frame_id in [-1, 1]:
            overlap_img = torch.zeros_like(ref_color)
            overlap_mask = torch.zeros_like(ref_mask)
            src_colors = org_prev_image if frame_id < 0 else org_next_image

            for cur_index in neighbor_cam_indices:
                # for partial surround view training
                src_color = src_colors[:, cur_index, ...]
                src_mask = mask[:, cur_index, ...]
                src_intrinsic = intrinsic[:, cur_index, ...]

                rel_pose = rel_pose_dict[(frame_id, cur_index)]
                warped_img, warped_mask = self.get_virtual_image(
                    src_color,
                    src_mask,
                    ref_depth,
                    src_intrinsic,
                    rel_pose,
                    ref_K,
                    ref_extrinsic,
                )

                warped_img = self.get_norm_image_single(
                    ref_color,
                    ref_mask,
                    warped_img,
                    warped_mask
                )

                # assuming no overlap between warped images
                overlap_img = overlap_img + warped_img
                overlap_mask = overlap_mask + warped_mask

            warped_views[('overlap', frame_id)] = overlap_img
            warped_views[('overlap_mask', frame_id)] = overlap_mask

        return warped_views
