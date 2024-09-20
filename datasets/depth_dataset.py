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

    def get_item(
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
        assert (
            len(prev_image_files)
            == len(cur_image_files)
            == len(next_image_files)
            == len(mask_files)
            == len(intrinsics)
            == len(extrinsics)
        )
        assert ref_extrinsic_idx < len(prev_image_files)

        prev_images = [self.load_image(file) for file in prev_image_files]
        cur_images = [self.load_image(file) for file in cur_image_files]
        next_images = [self.load_image(file) for file in next_image_files]
        masks = [self.load_mask(file) for file in mask_files]

        (
            aug_prev_images,
            aug_cur_images,
            aug_next_images,
            intrinsics,
            extrinsics,
            prev_images,
            cur_images,
            next_images,
        ) = self.sample_transforms(
            prev_images, cur_images, next_images, intrinsics, extrinsics,
        )
        masks = self.mask_transforms(masks)

        prev_images_tensor = torch.stack(prev_images)
        cur_images_tensor = torch.stack(cur_images)
        next_images_tensor = torch.stack(next_images)
        aug_prev_images_tensor = torch.stack(aug_prev_images)
        aug_cur_images_tensor = torch.stack(aug_cur_images)
        aug_next_images_tensor = torch.stack(aug_next_images)
        masks_tensor = torch.stack(masks)

        extrinsics_tensor = torch.from_numpy(np.stack(extrinsics))
        intrinsics_tensor = torch.stack([torch.eye(4) for _ in intrinsics])
        intrinsics_tensor[:, :3, :3] = torch.from_numpy(np.stack(intrinsics))

        ref_extrinsic_tensor = extrinsics_tensor[ref_extrinsic_idx:ref_extrinsic_idx + 1]

        return (
            aug_prev_images_tensor,
            aug_cur_images_tensor,
            aug_next_images_tensor,
            masks_tensor,
            intrinsics_tensor,
            extrinsics_tensor,
            ref_extrinsic_tensor,
            prev_images_tensor,
            cur_images_tensor,
            next_images_tensor,
        )

    def load_image(self, path: str):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def load_mask(self, path: str):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('L')
