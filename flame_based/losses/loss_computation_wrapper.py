from collections import defaultdict

import torch

from models.view_renderer import ViewRenderer


class LossComputationWrapper:
    def __init__(
        self,
        model,
        loss_fn,
        max_depth,
        min_depth,
        focal_length_scale,
        neighbor_cam_indices_map,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.focal_length_scale = focal_length_scale
        self.neighbor_cam_indices_map = neighbor_cam_indices_map
        self.view_renderer = ViewRenderer()
        self.loss_mean = {}

    def __call__(
        self,
        org_prev_images,
        org_cur_images,
        org_next_images,
        masks,
        depth_maps,
        intrinsics,
        extrinsics,
        prev_to_cur_poses,
        next_to_cur_poses,
    ):
        true_depth_maps = self.model.compute_true_depth_maps(
            depth_maps=depth_maps,
            intrinsic=intrinsics,
            max_depth=self.max_depth,
            min_depth=self.min_depth,
            focal_length_scale=self.focal_length_scale,
        )

        inv_extrinsics = torch.inverse(extrinsics)
        loss = 0
        loss_info = defaultdict(list)

        for cam_index in range(masks.shape[1]):
            neighbor_cam_indices = self.neighbor_cam_indices_map[cam_index]
            relative_poses = self.model.compute_relative_poses(
                cam_prev_to_cur_pose=prev_to_cur_poses[:, cam_index],
                cam_next_to_cur_pose=next_to_cur_poses[:, cam_index],
                cam_inv_extrinsic=inv_extrinsics[:, cam_index],
                extrinsic=extrinsics,
                neighbor_cam_indices=neighbor_cam_indices,
            )
            cam_warped_views = self.view_renderer(
                org_prev_image=org_prev_images,
                org_cur_image=org_cur_images,
                org_next_image=org_next_images,
                mask=masks,
                intrinsic=intrinsics,
                true_depth_map=true_depth_maps,
                prev_to_cur_pose=prev_to_cur_poses,
                next_to_cur_pose=next_to_cur_poses,
                cam_index=cam_index,
                neighbor_cam_indices=neighbor_cam_indices,
                rel_pose_dict=relative_poses,
                extrinsic=extrinsics,
            )
            cam_loss, loss_dict = self.loss_fn(
                cam_org_prev_image=org_prev_images[:, cam_index],
                cam_org_image=org_cur_images[:, cam_index],
                cam_org_next_image=org_next_images[:, cam_index],
                cam_target_view=cam_warped_views,
                cam_depth_map=depth_maps[:, cam_index],
                cam_mask=masks[:, cam_index],
            )
            loss = loss + cam_loss

            for k, v in loss_dict.items():
                loss_info[k].append(v)

        loss = loss / masks.shape[1]
        self.loss_mean = {k: sum(v) / len(v) for k, v in loss_info.items()}

        return loss
