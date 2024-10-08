import os
import tempfile
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from nuscenes.nuscenes import NuScenes
from PIL import Image
from pyquaternion import Quaternion
from torchvision import transforms

from .depth_dataset import DepthDataset
from .transforms import get_transforms


class NuScenesDataset(DepthDataset):
    def __init__(
        self,
        version: str = 'v1.0-mini',
        dataroot: str = '/data/sets/nuscenes',
        verbose: bool = True,
        cam_names: List[str] = [
            'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK',
        ],
        mask_dir: str = '',
        token_list_file: str = '',
        image_shape: Tuple[int, int] = (640, 352),  # (width, height)
        jittering: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),  # (brightness, contrast, saturation, hue)
        crop_train_borders: Tuple[float, float, float, float] = (),  # (left, top, right, down)
        crop_eval_borders: Tuple[float, float, float, float] = (),  # (left, top, right, down)
        ref_extrinsic_idx: int = 0,
        length: int = 0,
        gen_depth_map: bool = False,
        cache_dir: str = './.cache_nuscenes'
    ):
        self.dataset = NuScenes(
            version=version,
            dataroot=dataroot,
            verbose=verbose,
        )
        self.dataroot = Path(dataroot)
        self.mask_dir = Path(mask_dir)

        if not self.mask_dir.exists():
            raise FileNotFoundError(f'{self.mask_dir} not found.')

        token_list_file = Path(token_list_file)

        if not token_list_file.exists():
            raise FileNotFoundError(f'{token_list_file} not found.')

        with token_list_file.open() as f:
            token_list = [line.strip() for line in f]

        fd, ignored_file = tempfile.mkstemp(suffix='.txt', prefix='ignored_files', text=True)
        file_ignored = False

        self.token_list = []
        with os.fdopen(fd, mode='w') as f:
            for token in token_list:
                sample = self.dataset.get('sample', token)

                if sample['prev'] != '' and sample['next'] != '':
                    self.token_list.append(token)
                else:
                    file_ignored = True
                    f.write(f'{token}\n')

        if length > 0:
            self.token_list = self.token_list[:length]

        if file_ignored:
            warnings.warn(f'Ignored samples are listed in file {ignored_file}.')
        else:
            os.remove(ignored_file)

        # SUPPORTED_MODE = ['train', 'validation', 'test']
        mode = 'train'  # hard-coded to always resize input

        width, height = image_shape
        self._transforms = get_transforms(
            mode=mode,
            image_shape=(height, width),
            jittering=jittering,
            crop_train_borders=crop_train_borders,
            crop_eval_borders=crop_eval_borders,
        )
        self.cam_names = cam_names
        self.ref_extrinsic_idx = ref_extrinsic_idx
        self.gen_depth_map = gen_depth_map
        self.cache_dir = Path(cache_dir)
        super(NuScenesDataset, self).__init__(
            sample_transforms=self._sample_transforms,
            mask_transforms=self._mask_transforms,
        )

    def _sample_transforms(
        self,
        prev_images: List[Image.Image],
        cur_images: List[Image.Image],
        next_images: List[Image.Image],
        intrinsics: List[np.ndarray],
        extrinsics: List[np.ndarray],
    ):
        assert len(prev_images) == len(cur_images) == len(next_images) == len(intrinsics) == len(extrinsics)

        transformed_prev_images = []
        transformed_cur_images = []
        transformed_next_images = []
        transformed_intrinsics = []
        transformed_extrinsics = []
        original_prev_images = []
        original_cur_images = []
        original_next_images = []

        for prev_image, cur_image, next_image, intrinsic, extrinsic in zip(
            prev_images, cur_images, next_images, intrinsics, extrinsics,
        ):
            sample = {
                'rgb': cur_image,
                'rgb_context': [prev_image, next_image],
                'intrinsics': intrinsic,
                'extrinsics': extrinsic,
            }
            sample = self._transforms(sample)
            transformed_prev_images.append(sample['rgb_context'][0])
            transformed_next_images.append(sample['rgb_context'][1])
            transformed_cur_images.append(sample['rgb'])
            transformed_intrinsics.append(sample['intrinsics'])
            transformed_extrinsics.append(sample['extrinsics'])
            original_prev_images.append(sample['rgb_context_original'][0])
            original_next_images.append(sample['rgb_context_original'][1])
            original_cur_images.append(sample['rgb_original'])

        return (
            transformed_prev_images,
            transformed_cur_images,
            transformed_next_images,
            transformed_intrinsics,
            transformed_extrinsics,
            original_prev_images,
            original_cur_images,
            original_next_images,
        )

    def _mask_transforms(self, masks: List[Image.Image]):
        image_shape = self._transforms.keywords['image_shape']
        resize_transform = transforms.Resize(image_shape, interpolation=transforms.InterpolationMode.BILINEAR)
        tensor_transform = transforms.ToTensor()

        transformed_masks = []

        for mask in masks:
            mask = resize_transform(mask)
            mask = tensor_transform(mask)
            transformed_masks.append(mask)

        return transformed_masks

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        sample = self.dataset.get('sample', self.token_list[idx])

        prev_image_files = []
        cur_image_files = []
        next_image_files = []
        mask_files = []
        intrinsics = []
        extrinsics = []
        depths = []

        for cam_name in self.cam_names:
            cur_sample_data = self.dataset.get('sample_data', sample['data'][cam_name])
            cur_image_file = self.dataroot.joinpath(cur_sample_data['filename'])

            prev_sample_data = self.dataset.get('sample_data', cur_sample_data['prev'])
            prev_image_file = self.dataroot.joinpath(prev_sample_data['filename'])

            next_sample_data = self.dataset.get('sample_data', cur_sample_data['next'])
            next_image_file = self.dataroot.joinpath(next_sample_data['filename'])

            mask_file = self.mask_dir.joinpath(f'{cam_name}.png')

            cam_param = self.dataset.get('calibrated_sensor', cur_sample_data['calibrated_sensor_token'])
            intrinsic = np.array(cam_param['camera_intrinsic'], dtype=np.float32)
            extrinsic = Quaternion(cam_param['rotation']).transformation_matrix
            extrinsic[:3, 3] = np.array(cam_param['translation'])
            extrinsic = extrinsic.astype(np.float32)

            prev_image_files.append(prev_image_file)
            cur_image_files.append(cur_image_file)
            next_image_files.append(next_image_file)
            mask_files.append(mask_file)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)

            if self.gen_depth_map:
                depth = self.generate_depth_map(
                    sample,
                    cam_name,
                    cur_sample_data,
                )
                depths.append(torch.from_numpy(depth))

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
        ) = self.get_item(
            prev_image_files=prev_image_files,
            cur_image_files=cur_image_files,
            next_image_files=next_image_files,
            mask_files=mask_files,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ref_extrinsic_idx=self.ref_extrinsic_idx,
        )

        if self.gen_depth_map:
            for i, depth in enumerate(depths):
                _cur_image = self.load_image(cur_image_files[i])
                sample = {
                    'rgb': _cur_image,
                    'depth': depth,
                }
                sample = self._transforms(sample)
                depths[i] = sample['depth']

        # nuScenes only
        extrinsics = torch.inverse(extrinsics)
        ref_extrinsic = torch.inverse(ref_extrinsic)

        if not self.gen_depth_map:
            return (
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
            )
        else:
            depths_tensor = torch.stack(depths)
            return (
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
                depths_tensor,
            )

    def generate_depth_map(self, sample, sensor, cam_sample):
        """
        This function returns depth map for nuscenes dataset,
        result of depth map is saved in nuscenes/samples/DEPTH_MAP
        """
        # generate depth filename
        depth_file = self.cache_dir.joinpath(f'{cam_sample["filename"]}.npz')

        # load and return if exists
        if depth_file.exists():
            return np.load(str(depth_file), allow_pickle=True)['depth']
        else:
            lidar_sample = self.dataset.get('sample_data', sample['data']['LIDAR_TOP'])

            # lidar points
            lidar_file = self.dataroot.joinpath(lidar_sample['filename'])
            lidar_points = np.fromfile(lidar_file, dtype=np.float32)
            lidar_points = lidar_points.reshape(-1, 5)[:, :3]

            # lidar -> world
            lidar_pose = self.dataset.get('ego_pose', lidar_sample['ego_pose_token'])
            lidar_rotation = Quaternion(lidar_pose['rotation'])
            lidar_translation = np.array(lidar_pose['translation'])[:, None]
            lidar_to_world = np.vstack([
                np.hstack((lidar_rotation.rotation_matrix, lidar_translation)),
                np.array([0, 0, 0, 1]),
            ])

            # lidar -> ego
            sensor_sample = self.dataset.get('calibrated_sensor', lidar_sample['calibrated_sensor_token'])
            lidar_to_ego_rotation = Quaternion(sensor_sample['rotation']).rotation_matrix
            lidar_to_ego_translation = np.array(sensor_sample['translation']).reshape(1, 3)

            ego_lidar_points = np.dot(lidar_points[:, :3], lidar_to_ego_rotation.T)
            ego_lidar_points += lidar_to_ego_translation

            homo_ego_lidar_points = np.concatenate((ego_lidar_points, np.ones((ego_lidar_points.shape[0], 1))), axis=1)

            # world -> ego
            ego_pose = self.dataset.get('ego_pose', cam_sample['ego_pose_token'])
            ego_rotation = Quaternion(ego_pose['rotation']).inverse
            ego_translation = - np.array(ego_pose['translation'])[:, None]
            world_to_ego = np.vstack([
                np.hstack((ego_rotation.rotation_matrix, ego_rotation.rotation_matrix @ ego_translation)),
                np.array([0, 0, 0, 1]),
            ])

            # Ego -> sensor
            sensor_sample = self.dataset.get('calibrated_sensor', cam_sample['calibrated_sensor_token'])
            sensor_rotation = Quaternion(sensor_sample['rotation'])
            sensor_translation = np.array(sensor_sample['translation'])[:, None]
            sensor_to_ego = np.vstack([
                np.hstack((sensor_rotation.rotation_matrix, sensor_translation)),
                np.array([0, 0, 0, 1])
            ])
            ego_to_sensor = np.linalg.inv(sensor_to_ego)

            # lidar -> sensor
            lidar_to_sensor = ego_to_sensor @ world_to_ego @ lidar_to_world
            homo_ego_lidar_points = torch.from_numpy(homo_ego_lidar_points).float()
            cam_lidar_points = np.matmul(lidar_to_sensor, homo_ego_lidar_points.T).T

            # depth > 0
            depth_mask = cam_lidar_points[:, 2] > 0
            cam_lidar_points = cam_lidar_points[depth_mask]

            # sensor -> image
            intrinsics = np.eye(4)
            intrinsics[:3, :3] = sensor_sample['camera_intrinsic']
            pixel_points = np.matmul(intrinsics, cam_lidar_points.T).T
            pixel_points[:, :2] /= pixel_points[:, 2:3]

            # load image for pixel range
            image_filename = self.dataroot.joinpath(cam_sample['filename'])
            img = Image.open(image_filename)
            h, w, _ = np.array(img).shape

            # mask points in pixel range
            pixel_mask = (pixel_points[:, 0] >= 0) & (pixel_points[:, 0] <= w - 1)\
                & (pixel_points[:, 1] >= 0) & (pixel_points[:, 1] <= h - 1)
            valid_points = pixel_points[pixel_mask].round().int()
            valid_depth = cam_lidar_points[:, 2][pixel_mask]

            depth = np.zeros([h, w])
            depth[valid_points[:, 1], valid_points[:, 0]] = valid_depth

            # save depth map
            depth_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(str(depth_file), depth=depth)
            return depth
