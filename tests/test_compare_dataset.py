import os
import sys
from pathlib import Path

import numpy as np
import torch
from pyquaternion import Quaternion
from torchvision import transforms

sys.path.append(os.getcwd())
from datasets.nuscenes_dataset import NuScenesDataset  # noqa: E402

sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth')))
sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth', 'external', 'packnet_sfm')))
sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth', 'external', 'dgp')))
sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth', 'external', 'monodepth2')))
from external.dataset import stack_sample

from VFDepth import utils  # noqa: E402
from VFDepth.dataset.data_util import (align_dataset, img_loader,
                                       mask_loader_scene,
                                       transform_mask_sample)
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

    org_dataset = org_model.train_dataloader().dataset
    org_sample = org_dataset[0]
    # # print(org_dataset.dataset.filenames[0])
    # # get nuscenes dataset sample
    # idx = 0
    # frame_idx = org_dataset.filenames[idx].strip().split()[0]
    # sample_nusc = org_dataset.dataset.get('sample', frame_idx)

    # sample = []
    # contexts = []
    # if org_dataset.bwd:
    #     contexts.append(-1)
    # if org_dataset.fwd:
    #     contexts.append(1)

    # # loop over all cameras
    # org_rgb_imgs = []
    # for cam in org_dataset.cameras:
    #     cam_sample = org_dataset.dataset.get(
    #         'sample_data', sample_nusc['data'][cam])

    #     data = {
    #         'idx': idx,
    #         'sensor_name': cam,
    #         'contexts': contexts,
    #         'filename': cam_sample['filename'],
    #         'rgb': org_dataset.get_current('rgb', cam_sample),
    #         'intrinsics': org_dataset.get_current('intrinsics', cam_sample)
    #     }

    #     # if depth is returned
    #     if org_dataset.with_depth:
    #         data.update({
    #             'depth': org_dataset.generate_depth_map(sample_nusc, cam, cam_sample)
    #         })
    #     # if pose is returned
    #     if org_dataset.with_pose:
    #         data.update({
    #             'extrinsics': org_dataset.get_current('extrinsics', cam_sample)
    #         })
    #     # if mask is returned
    #     if org_dataset.with_mask:
    #         data.update({
    #             'mask': org_dataset.mask_loader(org_dataset.mask_path, '', cam)
    #         })
    #     # if context is returned
    #     if org_dataset.has_context:
    #         data.update({
    #             'rgb_context': org_dataset.get_context('rgb', cam_sample)
    #         })

    #     sample.append(data)

    # # apply same data transformations for all sensors
    # if org_dataset.data_transform:
    #     sample = [org_dataset.data_transform(smp) for smp in sample]
    #     sample = [transform_mask_sample(smp, org_dataset.data_transform) for smp in sample]
    #     org_rgb_imgs = [s['rgb'] for s in sample]
    #     org_rgb_original_imgs = [s['rgb_original'] for s in sample]
    # org_sample = sample
    # # # stack and align dataset for our trainer
    # # sample = stack_sample(sample)
    # # sample = align_dataset(sample, org_dataset.scales, contexts)

    dataset = NuScenesDataset(
        version='v1.0-mini',
        dataroot='../../datasets/nuscenes_mini',
        cam_names=[
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK',
        ],
        mask_dir='datasets/masks/nuscenes',
        token_list_file='datasets/file_lists/nuscenes/mini_train.txt',
        image_shape=(640, 352),
    )
    # print(dataset.token_list[0])
    (
        prev_images,
        cur_images,
        next_images,
        masks,
        intrinsics,
        extrinsics,
        ref_extrinsic,
        org_prev_images,
        org_cur_images,
        org_next_images,
    ) = dataset[0]
    ######################################################
    # sample = dataset.dataset.get('sample', dataset.token_list[idx])

    # prev_image_files = []
    # cur_image_files = []
    # next_image_files = []
    # mask_files = []
    # intrinsics = []
    # extrinsics = []

    # for cam_name in dataset.cam_names:
    #     cur_sample_data = dataset.dataset.get('sample_data', sample['data'][cam_name])
    #     cur_image_file = dataset.dataroot.joinpath(cur_sample_data['filename'])

    #     prev_sample_data = dataset.dataset.get('sample_data', cur_sample_data['prev'])
    #     prev_image_file = dataset.dataroot.joinpath(prev_sample_data['filename'])

    #     next_sample_data = dataset.dataset.get('sample_data', cur_sample_data['next'])
    #     next_image_file = dataset.dataroot.joinpath(next_sample_data['filename'])

    #     mask_file = dataset.mask_dir.joinpath(f'{cam_name}.png')

    #     cam_param = dataset.dataset.get('calibrated_sensor', cur_sample_data['calibrated_sensor_token'])
    #     intrinsic = np.array(cam_param['camera_intrinsic'], dtype=np.float32)
    #     extrinsic = Quaternion(cam_param['rotation']).transformation_matrix
    #     extrinsic[:3, 3] = np.array(cam_param['translation'])
    #     extrinsic = extrinsic.astype(np.float32)

    #     prev_image_files.append(prev_image_file)
    #     cur_image_files.append(cur_image_file)
    #     next_image_files.append(next_image_file)
    #     mask_files.append(mask_file)
    #     intrinsics.append(intrinsic)
    #     extrinsics.append(extrinsic)

    # (
    #     prev_images,
    #     cur_images,
    #     next_images,
    #     masks,
    #     intrinsics,
    #     extrinsics,
    #     ref_extrinsic,
    #     org_prev_images,
    #     org_cur_images,
    #     org_next_images,
    # ) = dataset.get_item(
    #     prev_image_files=prev_image_files,
    #     cur_image_files=cur_image_files,
    #     next_image_files=next_image_files,
    #     mask_files=mask_files,
    #     intrinsics=intrinsics,
    #     extrinsics=extrinsics,
    #     ref_extrinsic_idx=dataset.ref_extrinsic_idx,
    # )
    # prev_image_files = prev_image_files
    # cur_image_files = cur_image_files
    # next_image_files = next_image_files
    # mask_files = mask_files
    # intrinsics = intrinsics
    # extrinsics = extrinsics
    # ref_extrinsic_idx = dataset.ref_extrinsic_idx
    # assert all(intrinsic.shape == (3, 3) for intrinsic in intrinsics)
    # assert all(extrinsic.shape == (4, 4) for extrinsic in extrinsics)
    # assert (
    #     len(prev_image_files)
    #     == len(cur_image_files)
    #     == len(next_image_files)
    #     == len(mask_files)
    #     == len(intrinsics)
    #     == len(extrinsics)
    # )
    # assert ref_extrinsic_idx < len(prev_image_files)

    # prev_images = [dataset.load_image(file) for file in prev_image_files]
    # cur_images = [dataset.load_image(file) for file in cur_image_files]
    # next_images = [dataset.load_image(file) for file in next_image_files]
    # masks = [dataset.load_mask(file) for file in mask_files]

    # (
    #     prev_images,
    #     cur_images,
    #     next_images,
    #     aug_prev_images,
    #     aug_cur_images,
    #     aug_next_images,
    #     intrinsics,
    #     extrinsics,
    # ) = dataset.sample_transforms(
    #     prev_images, cur_images, next_images, intrinsics, extrinsics,
    # )
    # breakpoint()
    # for i, img in enumerate(cur_images):
    #     assert torch.all(abs(img - org_rgb_imgs[i]) < 1e-9)

    # masks = dataset.mask_transforms(masks)

    # prev_images_tensor = torch.stack(prev_images)
    # cur_images_tensor = torch.stack(cur_images)
    # next_images_tensor = torch.stack(next_images)
    # aug_prev_images_tensor = torch.stack(aug_prev_images)
    # aug_cur_images_tensor = torch.stack(aug_cur_images)
    # aug_next_images_tensor = torch.stack(aug_next_images)
    # masks_tensor = torch.stack(masks)

    # extrinsics_tensor = torch.from_numpy(np.stack(extrinsics))
    # intrinsics_tensor = torch.stack([torch.eye(4) for _ in intrinsics])
    # intrinsics_tensor[:, :3, :3] = torch.from_numpy(np.stack(intrinsics))

    # ref_extrinsic_tensor = extrinsics_tensor[ref_extrinsic_idx:ref_extrinsic_idx + 1]

    # prev_images = aug_prev_images_tensor,
    # cur_images = aug_cur_images_tensor,
    # next_images = aug_next_images_tensor,
    # masks = masks_tensor,
    # intrinsics = intrinsics_tensor,
    # extrinsics = extrinsics_tensor,
    # ref_extrinsic = ref_extrinsic_tensor,
    # org_prev_images = prev_images_tensor,
    # org_cur_images = cur_images_tensor,
    # org_next_images = next_images_tensor,
    #################################################
    # for i in range(len(org_rgb_imgs)):
    #     assert org_rgb_imgs[i].shape == cur_images[i].shape, (org_rgb_imgs[i].shape, cur_images[i].shape)
    #     print(abs(org_rgb_imgs[i] - cur_images[i]).max())

    # # nuScenes only
    # extrinsics = torch.inverse(extrinsics)
    # ref_extrinsic = torch.inverse(ref_extrinsic)

    assert prev_images.shape == org_sample['color_aug', -1, 0].shape
    assert cur_images.shape == org_sample['color_aug', 0, 0].shape
    assert next_images.shape == org_sample['color_aug', 1, 0].shape
    assert org_prev_images.shape == org_sample['color', -1, 0].shape
    assert org_cur_images.shape == org_sample['color', 0, 0].shape
    assert org_next_images.shape == org_sample['color', 1, 0].shape
    assert masks.shape == org_sample['mask'].shape
    assert intrinsics.shape == org_sample['K', 0].shape
    assert extrinsics.shape == torch.inverse(torch.from_numpy(org_sample['extrinsics'])).shape

    # assert sample['color_aug', -1, 0].shape == org_sample['color_aug', -1, 0].shape
    # assert sample['color_aug', 0, 0].shape == org_sample['color_aug', 0, 0].shape
    # assert sample['color_aug', 1, 0].shape == org_sample['color_aug', 1, 0].shape
    # assert sample['color', -1, 0].shape == org_sample['color', -1, 0].shape
    # assert sample['color', 0, 0].shape == org_sample['color', 0, 0].shape
    # assert sample['color', 1, 0].shape == org_sample['color', 1, 0].shape
    # assert sample['mask'].shape == org_sample['mask'].shape
    # assert sample['K', 0].shape == org_sample['K', 0].shape
    # assert torch.inverse(torch.from_numpy(sample['extrinsics'])).shape == torch.inverse(torch.from_numpy(org_sample['extrinsics'])).shape
    print('prev_images', abs(org_sample['color_aug', -1, 0] - org_sample['color_aug', -1, 0]).max())
    print('cur_images', abs(org_sample['color_aug', 0, 0] - org_sample['color_aug', 0, 0]).max())
    print('next_images', abs(org_sample['color_aug', 1, 0] - org_sample['color_aug', 1, 0]).max())
    print('org_prev_images', abs(org_sample['color', -1, 0] - org_sample['color', -1, 0]).max())
    print('org_cur_images', abs(org_sample['color', 0, 0] - org_sample['color', 0, 0]).max())
    print('org_next_images', abs(org_sample['color', 1, 0] - org_sample['color', 1, 0]).max())
    print('masks', abs(org_sample['mask'] - org_sample['mask']).max())
    print('intrinsics', abs(org_sample['K', 0] - org_sample['K', 0]).max())
    print('extrinsics', abs(torch.inverse(torch.from_numpy(org_sample['extrinsics'])) - torch.inverse(torch.from_numpy(org_sample['extrinsics']))).max())
