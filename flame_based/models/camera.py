from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Camera(ABC):
    def __init__(
        self,
        width: int,
        height: int,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
    ):
        grid_xy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        pix_coords = torch.stack(grid_xy, axis=0).view(2, height * width).T.contiguous()

        assert len(extrinsic.shape) == len(intrinsic.shape)
        assert extrinsic.shape[-2:] == (4, 4)
        assert intrinsic.shape[-2:] == (4, 4)

        self.width = width
        self.height = height
        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.pix_coords = pix_coords

    @abstractmethod
    def world_to_im(
        self,
        points_3d: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def im_to_cam(self, points_2d: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        pass

    def cam_to_world(self, points_3d: torch.Tensor):
        # points_3d (..., 3)
        assert points_3d.shape[-1] == 3

        inv_extrinsic = torch.inverse(self.extrinsic)

        points_3d = torch.cat([points_3d, torch.ones((*points_3d.shape[:-1], 1), device=points_3d.device)], dim=-1)
        points_3d = inv_extrinsic[..., :3, :] @ torch.transpose(points_3d, -2, -1)
        points_3d = torch.transpose(points_3d, -2, -1)

        # points_3d (..., 3)
        return points_3d

    def im_to_cam_map(self, depth_map: torch.Tensor) -> torch.Tensor:
        # depth_map (..., height, width, n)
        *pre_dims, height, width, n = depth_map.shape
        assert (height, width) == (self.height, self.width)

        pix_coords = self.pix_coords.to(depth_map.device)
        depth_map = depth_map.reshape(*pre_dims, height * width, n)

        points_3d = self.im_to_cam(pix_coords, depth_map)
        *pre_dims, num_pixels, _, _ = points_3d.shape
        assert num_pixels == height * width
        assert (n, 3) == points_3d.shape[-2:]

        points_3d = points_3d.reshape(*pre_dims, height, width, n, 3)

        # points_3d (..., height, width, n, 3)
        return points_3d


class PinHole(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        eps: float = 1e-8,
    ):
        super(PinHole, self).__init__(width, height, extrinsic, intrinsic)
        self.eps = eps

    def world_to_im(
        self,
        points_3d: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # points_3d (..., 3)
        assert points_3d.shape[-1] == 3

        points_3d = torch.cat([points_3d, torch.ones((*points_3d.shape[:-1], 1), device=points_3d.device)], dim=-1)
        points_3d = self.extrinsic[..., :3, :] @ torch.transpose(points_3d, -2, -1)
        points_3d = torch.transpose(points_3d, -2, -1)

        points_depth = points_3d[..., 2]
        is_point_front = points_depth > 0

        points_2d = self.intrinsic[..., :3, :3] @ torch.transpose(points_3d, -2, -1)
        points_2d = torch.transpose(points_2d, -2, -1)
        points_2d = points_2d[..., :2] / (points_2d[..., 2:3] + self.eps)
        points_2d = points_2d[..., :2]

        is_point_in_image = torch.logical_and(
            torch.logical_and(points_2d[..., 0] <= self.width - 1, points_2d[..., 0] >= 0),
            torch.logical_and(points_2d[..., 1] <= self.height - 1, points_2d[..., 1] >= 0),
        )

        valid_points = torch.logical_and(is_point_front, is_point_in_image)

        if normalize:
            points_2d = points_2d / torch.tensor([self.width - 1, self.height - 1], device=points_2d.device)
            points_2d = (points_2d - 0.5) * 2

        # points_2d (..., 2)
        # valid_points (..., )
        # points_depth (..., )
        return points_2d, valid_points, points_depth

    def im_to_cam(self, points_2d: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # points_2d (..., 2)
        # depth (..., n)
        assert points_2d.shape[-1] == 2
        points_2d = torch.cat([points_2d, torch.ones((*points_2d.shape[:-1], 1), device=points_2d.device)], dim=-1)
        inv_intrinsic = torch.inverse(self.intrinsic)

        points_3d = inv_intrinsic[..., :3, :3] @ torch.transpose(points_2d, -2, -1)
        points_3d = torch.transpose(points_3d, -2, -1)
        points_3d = points_3d.unsqueeze(-2) * depth.unsqueeze(-1)

        # points_3d (..., n, 3)
        return points_3d


class Fisheye(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        distortion: torch.Tensor,
        eps: float = 1e-8,
        max_count: int = 10,
    ):
        super(Fisheye, self).__init__(width, height, extrinsic, intrinsic)
        self.distortion = distortion
        self.fx = self.intrinsic[..., 0, 0]
        self.fy = self.intrinsic[..., 1, 1]
        self.cx = self.intrinsic[..., 0, 2]
        self.cy = self.intrinsic[..., 1, 2]
        self.eps = eps
        self.max_count = max_count

    def world_to_im(
        self,
        points_3d: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # points_3d (..., 3)
        assert points_3d.shape[-1] == 3

        points_3d = torch.cat([points_3d, torch.ones((*points_3d.shape[:-1], 1), device=points_3d.device)], dim=-1)
        points_3d = self.extrinsic[..., :3, :] @ torch.transpose(points_3d, -2, -1)
        points_3d = torch.transpose(points_3d, -2, -1)

        points_depth = points_3d[..., 2]
        is_point_front = points_depth > 0

        points_2d = points_3d[..., :2] / (points_3d[..., 2:3] + self.eps)
        r = torch.sqrt(points_2d[..., 0].unsqueeze(-2) @ points_2d[..., 1].unsqueeze(-1))
        theta = torch.arctan(r)
        theta_d = theta * (
            1 + self.distortion[0] * torch.pow(theta, 2)
            + self.distortion[1] * torch.pow(theta, 4)
            + self.distortion[2] * torch.pow(theta, 6)
            + self.distortion[3] * torch.pow(theta, 8)
        )

        inv_r = 1.0 / r if r > 1e-8 else 1.0
        cdist = theta_d * inv_r if r > 1e-8 else 1.0

        x = cdist * points_2d[..., 0]
        y = cdist * points_2d[..., 1]

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        points_2d = torch.stack([u, v], dim=-1)

        is_point_in_image = torch.logical_and(
            torch.logical_and(points_2d[..., 0] <= self.width - 1, points_2d[..., 0] >= 0),
            torch.logical_and(points_2d[..., 1] <= self.height - 1, points_2d[..., 1] >= 0),
        )

        valid_points = torch.logical_and(is_point_front, is_point_in_image)

        if normalize:
            points_2d = points_2d / torch.tensor([self.width - 1, self.height - 1], device=points_2d.device)
            points_2d = (points_2d - 0.5) * 2

        # points_2d (..., 2)
        # valid_points (..., )
        # points_depth (..., )
        return points_2d, valid_points, points_depth

    def im_to_cam(self, points_2d: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        u = points_2d[..., 0]
        v = points_2d[..., 1]

        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        theta_d = torch.sqrt(x ** 2 + y ** 2)
        theta_d == torch.min(torch.max(-torch.pi / 2, theta_d), torch.pi / 2)

        theta = theta_d

        for _ in range(self.max_count):
            theta2 = theta * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta2
            theta8 = theta6 * theta2
            k0_theta2 = self.distortion[0] * theta2
            k1_theta4 = self.distortion[1] * theta4
            k2_theta6 = self.distortion[2] * theta6
            k3_theta8 = self.distortion[3] * theta8
            theta_fix = (
                (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d)
                / (1 + 3 * k0_theta2 + 5 * k1_theta4 + 7 * k2_theta6 + 9 * k3_theta8)
            )
            theta = theta - theta_fix

            if torch.all(abs(theta_fix) < self.eps):
                break

        scale = torch.tan(theta) / theta_d

        theta_flipped = torch.logical_or(
            torch.logical_and(theta_d < 0, theta > 0),
            torch.logical_and(theta_d > 0, theta < 0),
        )

        x = x * scale
        y = y * scale
        z = torch.ones_like(x)
        z[theta_flipped] = -1e6

        points_3d = torch.stack([x, y, z], dim=-1)
        return points_3d


class VinAIFisheye(Fisheye):
    def __init__(self, camera_data_file: str):
        pass
