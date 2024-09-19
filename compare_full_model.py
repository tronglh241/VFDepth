import torch
from torch.utils.data import DataLoader

import utils
from flame_based.datasets.nuscenes_dataset import NuScenesDataset
from flame_based.losses.loss_computation_wrapper import LossComputationWrapper
from flame_based.losses.multi_cam_loss import MultiCamLoss
from flame_based.models.vf_depth import VFDepth
from flame_based.models.view_renderer import ViewRenderer
from models import VFDepthAlgo
from utils.misc import _NUSC_CAM_LIST, get_relcam

if __name__ == '__main__':
    cfg = utils.get_config('./configs/nuscenes/nusc_surround_fusion.yaml', mode='train')

    org_model = VFDepthAlgo(cfg, 0)
    org_model.load_weights()
    org_model.set_val()
    model = VFDepth()
    weight = torch.load('./output/weights/vfdepth.pt')
    model.load_state_dict(weight)
    model.cuda()
    model.eval()

    dataset = org_model.train_dataloader()
    inputs = next(iter(dataset))
    outputs, losses = org_model.process_batch(inputs, 0)
    breakpoint()
    # dataset = NuScenesDataset(
    #     version='v1.0-mini',
    #     dataroot='../VFDepth/input_data/nuscenes/',
    #     verbose=True,
    #     token_list_file='./dataset/nuscenes/val.txt',
    #     mask_dir='./dataset/nuscenes_mask',
    #     image_shape=(640, 352),
    #     jittering=(0.0, 0.0, 0.0, 0.0),
    #     crop_train_borders=(),
    #     crop_eval_borders=(),
    #     ref_extrinsic_idx=0,
    # )
    view_renderer = ViewRenderer()
    loss_fn = MultiCamLoss(
        disparity_smoothness=0.001,
        spatio_coeff=0.03,
        spatio_tempo_coeff=0.1,
    )
    loss_wrapper = LossComputationWrapper(
        model,
        loss_fn,
        max_depth=80,
        min_depth=1.5,
        focal_length_scale=300,
        neighbor_cam_indices_map=get_relcam(_NUSC_CAM_LIST),
    )

    # (
    #     prev_images,
    #     cur_images,
    #     next_images,
    #     masks,
    #     intrinsics,
    #     extrinsics,
    #     ref_extrinsic,
    #     aug_prev_images,
    #     aug_cur_images,
    #     aug_next_images,
    # ) = next(iter(DataLoader(dataset)))
    aug_prev_images = inputs['color_aug', -1, 0]
    aug_cur_images = inputs['color_aug', 0, 0]
    aug_next_images = inputs['color_aug', 1, 0]
    prev_images = inputs['color', -1, 0]
    cur_images = inputs['color', 0, 0]
    next_images = inputs['color', 1, 0]
    masks = inputs['mask']
    intrinsics = inputs['K', 0]
    extrinsics = inputs['extrinsics_inv']
    inv_extrinsics = inputs['extrinsics']
    ref_extrinsic = extrinsics[:, :1, ...]

    max_depth = 80.0
    min_depth = 1.5
    focal_length_scale = 300.0

    model = model.to('cuda')
    prev_images = prev_images.to('cuda')
    cur_images = cur_images.to('cuda')
    next_images = next_images.to('cuda')
    masks = masks.to('cuda')
    intrinsics = intrinsics.to('cuda')
    extrinsics = extrinsics.to('cuda')
    ref_extrinsic = ref_extrinsic.to('cuda')
    aug_prev_images = aug_prev_images.to('cuda')
    aug_cur_images = aug_cur_images.to('cuda')
    aug_next_images = aug_next_images.to('cuda')

    prev_to_cur_poses, next_to_cur_poses, depth_maps = model(
        prev_image=aug_prev_images,
        cur_image=aug_cur_images,
        next_image=aug_next_images,
        mask=masks,
        intrinsic=intrinsics,
        extrinsic=extrinsics,
        ref_extrinsic=ref_extrinsic,
    )
    true_depth_maps = model.compute_true_depth_maps(
        depth_maps=depth_maps,
        intrinsic=intrinsics,
        max_depth=max_depth,
        min_depth=min_depth,
        focal_length_scale=focal_length_scale,
    )
    inv_extrinsics = torch.inverse(extrinsics)
    loss = 0
    for cam in range(6):
        neighbor_cam_indices = get_relcam(_NUSC_CAM_LIST)[cam]
        relative_poses = model.compute_relative_poses(
            cam_prev_to_cur_pose=prev_to_cur_poses[:, cam],
            cam_next_to_cur_pose=next_to_cur_poses[:, cam],
            cam_inv_extrinsic=inv_extrinsics[:, cam],
            extrinsic=extrinsics,
            neighbor_cam_indices=neighbor_cam_indices,
        )
        cam_warped_views = view_renderer(
            prev_images,
            cur_images,
            next_images,
            masks,
            intrinsics,
            true_depth_maps,
            prev_to_cur_poses,
            next_to_cur_poses,
            cam,
            neighbor_cam_indices,
            relative_poses,
            extrinsics,
        )
        cam_loss, loss_dict = loss_fn(
            cam_org_prev_image=prev_images[:, cam],
            cam_org_image=cur_images[:, cam],
            cam_org_next_image=next_images[:, cam],
            cam_target_view=cam_warped_views,
            cam_depth_map=depth_maps[:, cam],
            cam_mask=masks[:, cam],
        )
        loss = loss + cam_loss
    loss = loss / 6
    print(loss)
    loss = loss_wrapper(
        prev_images,
        cur_images,
        next_images,
        masks,
        depth_maps,
        intrinsics,
        extrinsics,
        prev_to_cur_poses,
        next_to_cur_poses,
    )
    print(loss)
    outputs, losses = org_model.process_batch(inputs, 0)
    breakpoint()
