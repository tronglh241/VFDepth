from typing import Callable, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DepthDataset(Dataset):
    def __init__(
        self,
        sample_transforms: Callable,
        mask_transforms: Callable,
    ):
        self.sample_transforms = sample_transforms
        self.mask_transforms = mask_transforms

    def __getitem__(
        self,
        prev_image_files: List[str],
        cur_image_files: List[str],
        next_image_files: List[str],
        mask_files: List[str],
        intrinsics: List[np.ndarray],
        extrinsics: List[np.ndarray],
        ref_extrinsic_idx: int,
    ):
        assert all(intrinsic.shape == (3, 3) for intrinsic in intrinsics)
        assert all(extrinsic.shape == (4, 4) for extrinsic in extrinsics)

        prev_images = [self.load_image(file) for file in prev_image_files]
        cur_images = [self.load_image(file) for file in cur_image_files]
        next_images = [self.load_image(file) for file in next_image_files]
        masks = [self.load_mask(file) for file in mask_files]

        prev_images, cur_images, next_images, intrinsics, extrinsics = self.sample_transforms(
            prev_images, cur_images, next_images, intrinsics, extrinsics,
        )
        masks = self.mask_transforms(masks)
        prev_images = torch.stack(prev_images)
        cur_images = torch.stack(cur_images)
        next_images = torch.stack(next_images)
        masks = torch.stack(masks)

        extrinsics_tensor = torch.from_numpy(np.stack(extrinsics))
        intrinsics_tensor = torch.stack([torch.eye(4) for _ in intrinsics])
        intrinsics_tensor[:, :3, :3] = torch.from_numpy(np.stack(intrinsics))

        ref_extrinsic = extrinsics_tensor[ref_extrinsic_idx:ref_extrinsic_idx + 1]

        return prev_images, cur_images, next_images, masks, intrinsics_tensor, extrinsics_tensor, ref_extrinsic

    def load_image(self, path: str):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def load_mask(self, path: str):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('L')
