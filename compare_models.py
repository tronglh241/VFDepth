import torch
from torch.utils.data import DataLoader

import utils
from flame_based.datasets.nuscenes_dataset import NuScenesDataset
from flame_based.models.vf_depth import VFDepth
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

    with torch.no_grad():
        outputs, _ = org_model.process_batch(inputs, 0)
        prev_image = inputs[('color_aug', -1, 0)]
        cur_image = inputs[('color_aug', 0, 0)]
        next_image = inputs[('color_aug', 1, 0)]
        fusion_level = 2
        mask = inputs['mask']
        intrinsic = inputs['K', 0]
        # intrinsic = inputs['K', fusion_level + 1]
        # inv_intrinsic = inputs['inv_K', fusion_level + 1]
        extrinsic = inputs['extrinsics_inv']
        inv_extrinsic = inputs['extrinsics']
        ref_extrinsic = extrinsic[:, :1, ...]
        # ref_inv_extrinsic = inv_extrinsic[:, :1, ...]

        ######################################################################################
        prev_to_cur_poses, next_to_cur_poses, depth_maps = model(
            prev_image,
            cur_image,
            next_image,
            mask,
            intrinsic,
            extrinsic,
            # inv_intrinsic,
            # inv_extrinsic,
            ref_extrinsic,
            # ref_inv_extrinsic,
        )
        print(f'prev_image: {prev_image.shape}')
        print(f'cur_image: {cur_image.shape}')
        print(f'next_image: {next_image.shape}')
        print(f'mask: {mask.shape}')
        print(f'intrinsic: {intrinsic.shape}')
        print(f'extrinsic: {extrinsic.shape}')
        # print(f'inv_intrinsic: {inv_intrinsic.shape}')
        # print(f'inv_extrinsic: {inv_extrinsic.shape}')
        print(f'ref_extrinsic: {ref_extrinsic.shape}')
        # print(f'ref_inv_extrinsic: {ref_inv_extrinsic.shape}')
        print(f'prev_to_cur_poses: {prev_to_cur_poses.shape}')
        print(f'next_to_cur_poses: {next_to_cur_poses.shape}')
        print(f'depth_maps: {depth_maps.shape}')

        org_cam_T_cams = [outputs['cam', cam]['cam_T_cam', 0, -1] for cam in range(6)]

        for cam in range(6):
            assert torch.all(torch.abs(org_cam_T_cams[cam] - prev_to_cur_poses[:, cam]) < 1e-6)

        org_cam_T_cams = [outputs['cam', cam]['cam_T_cam', 0, 1] for cam in range(6)]

        for cam in range(6):
            assert torch.all(torch.abs(org_cam_T_cams[cam] - next_to_cur_poses[:, cam]) < 1e-6)

        for cam in range(6):
            assert torch.all(abs(depth_maps[:, cam] - outputs[('cam', cam)]['disp', 0]) < 1e-5)
        ######################################################################################

        for cam in range(6):
            org_rel_pose_dict = org_model.pose.compute_relative_cam_poses(inputs, outputs, cam)

            cam_prev_to_cur_pose = prev_to_cur_poses[:, cam]
            cam_next_to_cur_pose = next_to_cur_poses[:, cam]
            cam_inv_extrinsic = inv_extrinsic[:, cam]
            neighbor_cam_indices = get_relcam(_NUSC_CAM_LIST)[cam]
            relative_poses = model.compute_relative_poses(
                cam_prev_to_cur_pose,
                cam_next_to_cur_pose,
                cam_inv_extrinsic,
                extrinsic,
                neighbor_cam_indices,
            )

            assert len(org_rel_pose_dict) == len(relative_poses)
            for key in org_rel_pose_dict:
                assert torch.all(abs(org_rel_pose_dict[key] - relative_poses[key]) < 1e-6)
        ######################################################################################

        max_depth = 80.0
        min_depth = 1.5
        focal_length_scale = 300.0
        full_resolution_intrinsic = inputs['K', 0]
        true_depth_maps = model.compute_true_depth_maps(
            depth_maps,
            full_resolution_intrinsic,
            max_depth,
            min_depth,
            focal_length_scale,
        )

        for cam in range(6):
            assert torch.all(abs(outputs[('cam', cam)][('depth', 0)] - true_depth_maps[:, cam]) < 1e-2)
        ######################################################################################

    ##########################################################################################
    org_dataset = org_model._dataloaders['val'].dataset
    dataset = NuScenesDataset(
        version='v1.0-mini',
        dataroot='../VFDepth/input_data/nuscenes/',
        verbose=True,
        token_list_file='./dataset/nuscenes/val.txt',
        mask_dir='./dataset/nuscenes_mask',
        mode='train',
        image_shape=(640, 352),
        jittering=(0.0, 0.0, 0.0, 0.0),
        crop_train_borders=(),
        crop_eval_borders=(),
        ref_extrinsic_idx=0,
    )

    org_sample = next(iter(DataLoader(org_dataset)))
    (
        prev_images,
        cur_images,
        next_images,
        masks,
        intrinsics,
        extrinsics,
        inv_intrinsics,
        inv_extrinsics,
        ref_extrinsic,
        ref_inv_extrinsic,
    ) = next(iter(DataLoader(dataset)))

    assert torch.all(abs(org_sample['color_aug', 0, 0] - cur_images) < 1e-9)
    assert torch.all(abs(org_sample['color_aug', -1, 0] - prev_images) < 1e-9)
    assert torch.all(abs(org_sample['color_aug', 1, 0] - next_images) < 1e-9)
    assert torch.all(abs(org_sample['mask'] - masks) < 1e-9)
    assert torch.all(abs(org_sample['K', 0] - intrinsics) < 1e-9)
    assert torch.all(abs(org_sample['extrinsics'] - inv_extrinsics) < 1e-9)
    ##########################################################################################

    _NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']
    for key, ipt in org_sample.items():
        if key not in _NO_DEVICE_KEYS:
            if 'context' in key:
                org_sample[key] = [ipt[k].float().to(0) for k in range(len(org_sample[key]))]
            else:
                org_sample[key] = ipt.float().to(0)

    with torch.no_grad():
        outputs, _ = org_model.process_batch(org_sample, 0)

        # fusion_level = 2
        # intrinsics = org_sample['K', fusion_level + 1]
        # inv_intrinsics = org_sample['inv_K', fusion_level + 1]
        # extrinsics = org_sample['extrinsics_inv']
        # inv_extrinsics = org_sample['extrinsics']

        prev_to_cur_poses, next_to_cur_poses, depth_maps = model(
            prev_images.cuda(),
            cur_images.cuda(),
            next_images.cuda(),
            masks.cuda(),
            intrinsics.cuda(),
            extrinsics.cuda(),
            # inv_intrinsics.cuda(),
            # inv_extrinsics.cuda(),
            ref_extrinsic.cuda(),
            # ref_inv_extrinsic.cuda(),
        )

        for cam in range(6):
            assert torch.all(abs(depth_maps[:, cam] - outputs[('cam', cam)]['disp', 0]) < 1e-5)
