from abc import ABC, abstractmethod
from typing import Tuple

import torch


class Camera(ABC):
    def __init__(self, width: int, height: int):
        grid_xy = torch.meshgrid(torch.arange(width), torch.arange(height), indexing='xy')
        pix_coords = torch.stack(grid_xy, axis=0).view(2, height * width).T.contiguous()

        self.width = width
        self.height = height
        self.pix_coords = pix_coords

    @abstractmethod
    def project(
        self,
        points_3d: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def inv_project(self, points_2d: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        pass

    def inv_project_im(self, depth_map: torch.Tensor) -> torch.Tensor:
        return self.inv_project(self.pix_coords.to(depth_map.device), depth_map)


class PinHole(Camera):
    def __init__(
        self,
        width: int,
        height: int,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        eps: float = 1e-8,
    ):
        super(PinHole, self).__init__(width, height)

        assert len(extrinsic.shape) == len(intrinsic.shape)
        assert extrinsic.shape[-2:] == (4, 4)
        assert intrinsic.shape[-2:] == (4, 4)

        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.eps = eps

    def project(
        self,
        points_3d: torch.Tensor,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return points_2d, valid_points, points_depth

    def project_org(self, points_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert points_3d.shape[-1] == 3

        points_3d = torch.cat([points_3d, torch.ones((*points_3d.shape[:-1], 1), device=points_3d.device)], dim=-1)
        assert points_3d.shape[-1] == 4
        points_3d = self.extrinsic[..., :3, :] @ torch.transpose(points_3d, -2, -1)
        points_3d = torch.transpose(points_3d, -2, -1)
        assert points_3d.shape[-1] == 3

        points_depth = points_3d[..., 2]
        is_point_front = points_depth > 0
        assert is_point_front.shape == points_3d.shape[:-1]

        points_2d = self.intrinsic[..., :3, :3] @ torch.transpose(points_3d, -2, -1)
        points_2d = torch.transpose(points_2d, -2, -1)
        assert points_2d.shape[-1] == 3
        points_2d = points_2d[..., :2] / (points_2d[..., 2:3] + self.eps)
        if not torch.all(torch.isfinite(points_2d)):
            pix_coords = torch.clamp(points_2d, min=-self.width * 2, max=self.width * 2)
        else:
            pix_coords = points_2d
        # print(points_2d[..., :2].mean(), points_2d[..., :2].min(), points_2d[..., :2].max())
        # print(points_2d[..., :2].mean(), points_2d[..., :2].min(), points_2d[..., :2].max())
        # TODO
        # is_point_in_image = torch.logical_and(
        #     torch.logical_and(points_2d[..., 0] <= self.width - 1, points_2d[..., 0] >= 0),
        #     torch.logical_and(points_2d[..., 1] <= self.height - 1, points_2d[..., 1] >= 0),
        # )
        pix_coords = pix_coords / torch.tensor([self.width - 1, self.height - 1], device=pix_coords.device)
        pix_coords = (pix_coords - 0.5) * 2
        # print(pix_coords.mean())
        is_point_in_image = ~(torch.logical_or(pix_coords > 1, pix_coords < -1).sum(dim=-1) > 0)
        # print(is_point_in_image.sum())

        points_2d = points_2d[..., :2]
        assert points_2d.shape == (*points_3d.shape[:-1], 2)
        # breakpoint()
        valid_points = torch.logical_and(is_point_front, is_point_in_image)

        return points_2d, valid_points, points_depth

    def inv_project(self, points_2d: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # points_2d (..., 2)
        # depth (..., n)
        points_2d = torch.cat([points_2d, torch.ones((*points_2d.shape[:-1], 1), device=points_2d.device)], dim=-1)
        inv_intrinsic = torch.inverse(self.intrinsic)

        points_3d = inv_intrinsic[..., :3, :3] @ torch.transpose(points_2d, -2, -1)
        points_3d = torch.transpose(points_3d, -2, -1)
        points_3d = points_3d.unsqueeze(-2) * depth.unsqueeze(-1)

        # points_3d (..., n, 2)
        return points_3d
