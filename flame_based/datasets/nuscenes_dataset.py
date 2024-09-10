from pathlib import Path
from typing import List, Tuple

import numpy as np
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
        mode: str = 'train',
        image_shape: Tuple[int, int] = (640, 352),  # (width, height)
        jittering: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),  # (brightness, contrast, saturation, hue)
        crop_train_borders: Tuple[float, float, float, float] = (),  # (left, top, right, down)
        crop_eval_borders: Tuple[float, float, float, float] = (),  # (left, top, right, down)
        ref_extrinsic_idx: int = 0,
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
            self.token_list = [line.strip() for line in f]

        for token in self.token_list:
            sample = self.dataset.get('sample', token)
            assert sample['prev'] != '', f'Sample {token} does not have a predecessor.'
            assert sample['next'] != '', f'Sample {token} does not have a successor.'

        SUPPORTED_MODE = ['train', 'validation', 'test']
        if mode not in SUPPORTED_MODE:
            raise ValueError(f'`mode` has to be in {SUPPORTED_MODE}.')

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
        super(NuScenesDataset, self).__init__(
            sample_transforms=self._sample_transforms,
            mask_transforms=self._mask_transforms,
        )

    def _sample_transforms(self, prev_images, cur_images, next_images, intrinsics, extrinsics):
        assert len(prev_images) == len(cur_images) == len(next_images) == len(intrinsics) == len(extrinsics)

        transformed_prev_images = []
        transformed_cur_images = []
        transformed_next_images = []
        transformed_intrinsics = []
        transformed_extrinsics = []

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

        return (
            transformed_prev_images,
            transformed_cur_images,
            transformed_next_images,
            transformed_intrinsics,
            transformed_extrinsics,
        )

    def _mask_transforms(self, masks):
        image_shape = self._transforms.keywords['image_shape']
        resize_transform = transforms.Resize(image_shape, interpolation=Image.ANTIALIAS)
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

        return super(NuScenesDataset, self).__getitem__(
            prev_image_files=prev_image_files,
            cur_image_files=cur_image_files,
            next_image_files=next_image_files,
            mask_files=mask_files,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            ref_extrinsic_idx=self.ref_extrinsic_idx,
        )
