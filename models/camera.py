from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Camera(ABC):
    def __init__(
        self,
        width: int,
        height: int,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        inv_intrinsic: torch.Tensor = None,
        inv_extrinsic: torch.Tensor = None,
        fov: torch.Tensor = None,
    ):
        grid_xy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        pix_coords = torch.stack(grid_xy, dim=0).view(2, height * width).T.contiguous()

        assert len(extrinsic.shape) == len(intrinsic.shape)
        assert intrinsic.shape[-2:] == (4, 4), intrinsic.shape[-2:]
        assert extrinsic.shape[-2:] == (4, 4), extrinsic.shape[-2:]

        if inv_intrinsic is not None:
            assert inv_intrinsic.shape[-2:] == (4, 4), inv_intrinsic.shape[-2:]
        else:
            inv_intrinsic = torch.inverse(intrinsic)

        if inv_extrinsic is not None:
            assert inv_extrinsic.shape[-2:] == (4, 4), inv_extrinsic.shape[-2:]
        else:
            inv_extrinsic = torch.inverse(extrinsic)

        if fov is None:
            self.fov = torch.Tensor([
                [-torch.pi / 2, torch.pi / 2],
                [-torch.pi / 2, torch.pi / 2],
            ])
        else:
            self.fov = fov

        self.width = width
        self.height = height
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.inv_intrinsic = inv_intrinsic
        self.inv_extrinsic = inv_extrinsic
        self.pix_coords = pix_coords

    def world_to_im(
        self,
        points_3d: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # points_3d (..., 3)
        assert points_3d.shape[-1] == 3

        points_3d = self.world_to_cam(points_3d)
        pre_dims = points_3d.shape[:-1]

        is_in_fov = self.in_fov(points_3d)

        points_depth = points_3d[..., 2]
        is_front = points_depth > 0
        assert pre_dims == is_front.shape

        points_2d = self.cam_to_im(points_3d)
        assert points_2d.shape == (*pre_dims, 2)
        is_in_image = self.is_in_image(points_2d)
        assert pre_dims == is_in_image.shape

        valid_points = torch.logical_and(
            is_in_fov,
            torch.logical_and(is_front, is_in_image),
        )
        assert pre_dims == valid_points.shape

        # TODO: this might not be necessary
        if not torch.all(torch.isfinite(points_2d)):
            points_2d = torch.clamp(points_2d, min=-self.width * 2, max=self.width * 2)

        if normalize:
            # TODO: use the below line instead of the two next
            # points_2d = points_2d / torch.tensor([self.width - 1, self.height - 1], device=points_2d.device)
            points_2d[..., 0] = points_2d[..., 0] / (self.width - 1)
            points_2d[..., 1] = points_2d[..., 1] / (self.height - 1)
            points_2d = (points_2d - 0.5) * 2

        # points_2d (..., 2)
        # valid_points (..., )
        # points_depth (..., )
        return points_2d, valid_points, points_depth

    @abstractmethod
    def cam_to_im(self, points_3d: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def im_to_cam(self, points_2d: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def cam_to_world(self, points_3d: torch.Tensor):
        # points_3d (..., 3)
        assert points_3d.shape[-1] == 3

        points_3d = torch.cat([points_3d, torch.ones((*points_3d.shape[:-1], 1), device=points_3d.device)], dim=-1)
        points_3d = self.inv_extrinsic[..., :3, :] @ torch.transpose(points_3d, -2, -1)
        points_3d = torch.transpose(points_3d, -2, -1)

        # points_3d (..., 3)
        return points_3d

    def world_to_cam(self, points_3d: torch.Tensor) -> torch.Tensor:
        points_3d = torch.cat([points_3d, torch.ones((*points_3d.shape[:-1], 1), device=points_3d.device)], dim=-1)
        points_3d = self.extrinsic[..., :3, :] @ torch.transpose(points_3d, -2, -1)
        points_3d = torch.transpose(points_3d, -2, -1)

        return points_3d

    def im_to_cam_map(self, depth_map: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # depth_map (..., height, width, n)
        # mask (..., 1, height, width)
        *pre_dims, height, width, n = depth_map.shape
        assert (height, width) == (self.height, self.width)
        assert (1, height, width) == mask.shape[-3:]

        pix_coords = self.pix_coords.to(depth_map.device)
        depth_map = depth_map.reshape(*pre_dims, height * width, n)

        points_3d, valid_points = self.im_to_cam(pix_coords, depth_map)
        *pre_dims, num_pixels, _, _ = points_3d.shape
        assert num_pixels == height * width
        assert (n, 3) == points_3d.shape[-2:]

        points_3d = points_3d.reshape(*pre_dims, height, width, n, 3)
        valid_points = valid_points.reshape(*pre_dims, height, width)
        valid_points = torch.where(mask.squeeze(1) > 0.5, valid_points, False)

        # points_3d (..., height, width, n, 3)
        # valid_points (..., height, width)
        assert valid_points.shape == points_3d.shape[:-2]
        return points_3d, valid_points

    def is_in_image(self, points_2d: torch.Tensor) -> torch.Tensor:
        is_point_in_image = torch.logical_and(
            torch.logical_and(points_2d[..., 0] <= self.width - 1, points_2d[..., 0] >= 0),
            torch.logical_and(points_2d[..., 1] <= self.height - 1, points_2d[..., 1] >= 0),
        )
        return is_point_in_image

    def _in_fov(self, x1: torch.Tensor, x2: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
        fov = fov.to(x1.device)
        return torch.logical_and(fov[..., 0:1] <= torch.arctan2(x1, x2), torch.arctan2(x1, x2) <= fov[..., 1:2])

    def in_fov(self, points_3d: torch.Tensor) -> torch.Tensor:
        return torch.logical_and(
            self._in_fov(points_3d[..., 0], points_3d[..., 2], self.fov[..., 0, :]),  # horizontal
            self._in_fov(points_3d[..., 1], points_3d[..., 2], self.fov[..., 1, :]),  # vertical
        )


class PinHole(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        inv_intrinsic: torch.Tensor = None,
        inv_extrinsic: torch.Tensor = None,
        fov: torch.Tensor = None,
        eps: float = 1e-8,
    ):
        super(PinHole, self).__init__(
            width=width,
            height=height,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            inv_intrinsic=inv_intrinsic,
            inv_extrinsic=inv_extrinsic,
            fov=fov,
        )
        self.eps = eps

    def cam_to_im(
        self,
        points_3d: torch.Tensor,
    ) -> torch.Tensor:
        # points_3d (..., 3)
        assert points_3d.shape[-1] == 3

        points_2d = self.intrinsic[..., :3, :3] @ torch.transpose(points_3d, -2, -1)
        points_2d = torch.transpose(points_2d, -2, -1)
        points_2d = points_2d[..., :2] / (points_2d[..., 2:3] + self.eps)
        points_2d = points_2d[..., :2]

        return points_2d

    def im_to_cam(self, points_2d: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # points_2d (..., 2)
        # depth (..., n)
        assert points_2d.shape[-1] == 2
        points_2d = torch.cat([points_2d, torch.ones((*points_2d.shape[:-1], 1), device=points_2d.device)], dim=-1)

        points_3d = self.inv_intrinsic[..., :3, :3] @ torch.transpose(points_2d, -2, -1)
        points_3d = torch.transpose(points_3d, -2, -1)
        points_3d = points_3d.unsqueeze(-2) * depth.unsqueeze(-1)

        valid_points = torch.ones((points_3d.shape[:-2]), dtype=torch.bool, device=points_2d.device)
        # points_3d (..., n, 3)
        return points_3d, valid_points


class Fisheye(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        intrinsic: torch.Tensor,
        extrinsic: torch.Tensor,
        distortion: torch.Tensor,
        fov: torch.Tensor = None,
        eps: float = 1e-8,
        max_count: int = 10,
    ):
        super(Fisheye, self).__init__(
            width=width,
            height=height,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            fov=fov,
        )
        self.distortion = distortion
        self.fx = self.intrinsic[..., 0:1, 0]
        self.fy = self.intrinsic[..., 1:2, 1]
        self.cx = self.intrinsic[..., 0:1, 2]
        self.cy = self.intrinsic[..., 1:2, 2]
        self.eps = eps
        self.max_count = max_count

    def cam_to_im(
        self,
        points_3d: torch.Tensor,
    ) -> torch.Tensor:
        # points_3d (..., 3)
        assert points_3d.shape[-1] == 3

        points_2d = points_3d[..., :2] / (points_3d[..., 2:3] + self.eps)
        r = torch.sqrt(points_2d.unsqueeze(-2) @ points_2d.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        assert points_2d.shape[:-1] == r.shape, (points_2d.shape[:-1], r.shape)

        theta = torch.arctan(r)
        theta_d = theta * (
            1 + self.distortion[..., 0:1] * torch.pow(theta, 2)
            + self.distortion[..., 1:2] * torch.pow(theta, 4)
            + self.distortion[..., 2:3] * torch.pow(theta, 6)
            + self.distortion[..., 3:4] * torch.pow(theta, 8)
        )
        assert theta_d.shape == points_3d.shape[:-1], (theta_d.shape, points_3d.shape[:-1])
        inv_r = torch.where(r > 1e-8, 1.0 / r, 1.0)
        cdist = torch.where(r > 1e-8, theta_d * inv_r, 1.0)

        x = cdist * points_2d[..., 0]
        y = cdist * points_2d[..., 1]

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy

        points_2d = torch.stack([u, v], dim=-1)
        return points_2d

    def im_to_cam(self, points_2d: torch.Tensor, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        u = points_2d[..., 0]
        v = points_2d[..., 1]

        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy

        theta_d = torch.sqrt(x ** 2 + y ** 2)
        valid_points = torch.logical_and(
            theta_d < torch.pi / 2,
            theta_d > -torch.pi / 2,
        )
        theta_d = torch.clamp(theta_d, -torch.pi / 2, torch.pi / 2)

        theta = theta_d

        for _ in range(self.max_count):
            theta2 = theta * theta
            theta4 = theta2 * theta2
            theta6 = theta4 * theta2
            theta8 = theta6 * theta2
            k0_theta2 = self.distortion[..., 0:1] * theta2
            k1_theta4 = self.distortion[..., 1:2] * theta4
            k2_theta6 = self.distortion[..., 2:3] * theta6
            k3_theta8 = self.distortion[..., 3:4] * theta8
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
        valid_points = torch.logical_and(
            valid_points,
            ~theta_flipped,
        )

        x = x * scale
        y = y * scale
        z = torch.ones_like(x)
        z[theta_flipped] = -1e6

        points_3d = torch.stack([x, y, z], dim=-1)
        points_3d = points_3d.unsqueeze(-2) * depth.unsqueeze(-1)
        return points_3d, valid_points
