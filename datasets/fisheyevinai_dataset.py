import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL.Image import Image
from torchvision import transforms

from .depth_dataset import DepthDataset
from .transforms import get_transforms


class FisheyeVinAIDataset(DepthDataset):
    def __init__(
        self,
        dataroot: str,
        mask_dir: str = '',
        calib_file: str = '',
        token_list_file: str = '',
        cam_names: List[str] = ['front', 'left', 'rear', 'right'],
        fovs: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        image_shape: Tuple[int, int] = (640, 400),  # (width, height)
        jittering: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),  # (brightness, contrast, saturation, hue)
        crop_train_borders: Tuple[float, float, float, float] = (),  # (left, top, right, down)
        crop_eval_borders: Tuple[float, float, float, float] = (),  # (left, top, right, down)
        ref_extrinsic_idx: int = 0,
        length: int = 0,
    ):
        self.dataroot = Path(dataroot)
        self.mask_dir = Path(mask_dir)

        if not self.mask_dir.exists():
            raise FileNotFoundError(f'{self.mask_dir} not found.')

        token_list_file = Path(token_list_file)

        if not token_list_file.exists():
            raise FileNotFoundError(f'{token_list_file} not found.')

        with token_list_file.open() as f:
            self.token_list = [line.strip() for line in f]

        if length > 0:
            self.token_list = self.token_list[:length]

        self.calib_info = self.load_calib(calib_file)

        # SUPPORTED_MODE = ['train', 'validation', 'test']
        mode = 'train'  # hard-coded to always resize input

        width, height = image_shape
        self.cam_names = cam_names
        self.ref_extrinsic_idx = ref_extrinsic_idx
        self._transforms = get_transforms(
            mode=mode,
            image_shape=(height, width),
            jittering=jittering,
            crop_train_borders=crop_train_borders,
            crop_eval_borders=crop_eval_borders,
        )

        for token in self.token_list:
            for cam_name in self.cam_names:
                assert self.dataroot.joinpath(
                    token, f'{cam_name}_current.jpg'
                ).exists(), f'Files not found, token {token}.'
                assert self.dataroot.joinpath(
                    token, f'{cam_name}_prev.jpg'
                ).exists(), f'Files not found, token {token}.'
                assert self.dataroot.joinpath(
                    token, f'{cam_name}_next.jpg'
                ).exists(), f'Files not found, token {token}.'

        if fovs is None:
            self.fovs = {
                cam_name: torch.Tensor([
                    [-torch.pi / 2, torch.pi / 2],
                    [-torch.pi / 2, torch.pi / 2],
                ])
                for cam_name in cam_names
            }
        else:
            self.fovs = {k: torch.Tensor(v) for k, v in fovs.items()}

        super(FisheyeVinAIDataset, self).__init__(
            sample_transforms=self._sample_transforms,
            mask_transforms=self._mask_transforms,
        )

    def _sample_transforms(
        self,
        prev_images: List[Image],
        cur_images: List[Image],
        next_images: List[Image],
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

    def _mask_transforms(self, masks):
        image_shape = self._transforms.keywords['image_shape']
        resize_transform = transforms.Resize(image_shape, interpolation=transforms.InterpolationMode.BILINEAR)
        tensor_transform = transforms.ToTensor()

        transformed_masks = []

        for mask in masks:
            mask = resize_transform(mask)
            mask = tensor_transform(mask)
            transformed_masks.append(mask)

        return transformed_masks

    def load_calib(self, cam_data_file):
        cam_mapping = {
            'left': 0,
            'front': 1,
            'rear': 2,
            'right': 3,
        }

        with open(cam_data_file) as f:
            cam_data = json.load(f)
            cam_data = {data['camPos']: data for data in cam_data['Items']}

        calib_info = {}

        for cam_name in cam_mapping:
            pos = cam_mapping[cam_name]
            data = cam_data[pos]
            R = np.array(data['matrixR']).reshape(3, 3)
            T = np.array([data['vectT']])
            extrinsic = np.concatenate((R, T.T), axis=1)
            extrinsic = np.concatenate((extrinsic, np.zeros((1, 4))), axis=0)
            extrinsic[-1, -1] = 1.0

            K = np.array(data['matrixK'])
            intrinsic = np.zeros((3, 3))
            intrinsic[0, 0] = K[0]
            intrinsic[0, 2] = K[1]
            intrinsic[1, 1] = K[2]
            intrinsic[1, 2] = K[3]
            intrinsic[2, 2] = 1.0

            distortion = data['matrixD']

            extrinsic = torch.Tensor(extrinsic)
            intrinsic = torch.Tensor(intrinsic)
            distortion = torch.Tensor(distortion)

            calib_info[cam_name] = {
                'extrinsic': extrinsic,
                'intrinsic': intrinsic,
                'distortion': distortion,
            }

        return calib_info

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx):
        token = self.token_list[idx]

        prev_image_files = []
        cur_image_files = []
        next_image_files = []
        mask_files = []
        intrinsics = []
        extrinsics = []
        distortions = []
        fovs = []

        for cam_name in self.cam_names:
            cur_image_file = self.dataroot.joinpath(token, f'{cam_name}_current.jpg')
            prev_image_file = self.dataroot.joinpath(token, f'{cam_name}_prev.jpg')
            next_image_file = self.dataroot.joinpath(token, f'{cam_name}_next.jpg')
            mask_file = self.mask_dir.joinpath(f'{cam_name}.png')

            intrinsic = self.calib_info[cam_name]['intrinsic']
            extrinsic = self.calib_info[cam_name]['extrinsic']
            distortion = self.calib_info[cam_name]['distortion']
            fov = self.fovs[cam_name]

            prev_image_files.append(prev_image_file)
            cur_image_files.append(cur_image_file)
            next_image_files.append(next_image_file)
            mask_files.append(mask_file)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            distortions.append(distortion)
            fovs.append(fov)

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

        distortions_tensor = torch.stack(distortions)
        fovs_tensor = torch.stack(fovs)

        return (
            prev_images,
            cur_images,
            next_images,
            masks,
            intrinsics,
            extrinsics,
            distortions_tensor,
            ref_extrinsic,
            org_prev_images,
            org_cur_images,
            org_next_images,
            fovs_tensor,
        )
