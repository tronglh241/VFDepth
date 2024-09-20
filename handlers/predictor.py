from pathlib import Path
from typing import Any, Callable

import open3d as o3d
import torch
from flame.handlers import Handler
from ignite.engine import Events
from ignite.handlers import global_step_from_engine
from torchvision import transforms

from models.camera import Fisheye, PinHole


class Predictor(Handler):
    def __init__(
        self,
        trainer,
        evaluator,
        compute_true_depth_maps,
        out_dir: str,
        out_name: str,
        depth_limit: float = None,
        output_transform: Callable = lambda x: x,
        event_name: Any = Events.ITERATION_COMPLETED,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_name = out_name

        self.output_transform = output_transform
        self.global_step_from_engine = global_step_from_engine(trainer, event_name)

        actions = []

        action = {
            'engine': evaluator,
            'event': Events.ITERATION_COMPLETED,
            'func': self.save_pcd,
        }
        actions.append(action)

        action = {
            'engine': evaluator,
            'event': Events.EPOCH_STARTED,
            'func': self.reset,
        }
        actions.append(action)

        self.to_image = transforms.ToPILImage()
        self.compute_true_depth_maps = compute_true_depth_maps
        self.depth_limit = depth_limit
        self.cnt = 0
        super(Predictor, self).__init__(actions=actions)

    def reset(self):
        self.cnt = 0

    def save_pcd(self, engine):
        (
            depth_maps,
            images,
            masks,
            intrinsics,
            extrinsics,
            distortions,
            fovs,
        ) = self.output_transform(engine.state.output)
        assert len(depth_maps.shape) == 5

        true_depth_maps = self.compute_true_depth_maps(depth_maps, intrinsics)
        batch_size, num_cams, _, height, width = images.shape

        for fid in range(batch_size):
            if distortions is None:
                camera = PinHole(
                    width=width,
                    height=height,
                    extrinsic=extrinsics[fid],
                    intrinsic=intrinsics[fid],
                    fov=fovs[fid] if fovs is not None else None,
                )
            else:
                camera = Fisheye(
                    width=width,
                    height=height,
                    extrinsic=extrinsics[fid],
                    intrinsic=intrinsics[fid],
                    distortion=distortions[fid] if distortions is not None else None,
                    fov=fovs[fid] if fovs is not None else None,
                )

            points_3d, valid_points_3d = camera.im_to_cam_map(
                true_depth_maps[fid].permute(0, 2, 3, 1),
                masks[fid],
            )
            # points_3d (num_cams, height, width, 1, 3)
            assert points_3d.shape[-2] == 1

            points_3d = points_3d.squeeze(-2)
            points_3d = points_3d.view(num_cams, height * width, 3)
            points_3d = camera.cam_to_world(points_3d)
            valid_points_3d = valid_points_3d.view(num_cams, height * width)
            colors = images[fid].permute(0, 2, 3, 1).view(num_cams, height * width, 3)
            pcd_combined = o3d.geometry.PointCloud()

            for cam_idx, (points, point_colors, valid_pos) in enumerate(zip(points_3d, colors, valid_points_3d)):
                points = points[valid_pos]
                point_colors = point_colors[valid_pos]

                if self.depth_limit is not None:
                    depth = torch.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2)
                    points = points[depth < self.depth_limit]
                    point_colors = point_colors[depth < self.depth_limit]

                positions = o3d.utility.Vector3dVector(points.cpu().numpy())
                colors = o3d.utility.Vector3dVector(point_colors.cpu().numpy())
                pcd = o3d.geometry.PointCloud(positions)
                pcd.colors = colors

                pcd_combined += pcd

                # im_file = self.out_dir.joinpath(f'{self.out_name}_{fid:06d}_image_cam{cam_idx}.jpg')
                # mask_file = self.out_dir.joinpath(f'{self.out_name}_{fid:06d}_mask_cam{cam_idx}.png')

                # self.to_image(images[fid, cam_idx]).save(im_file)
                # self.to_image(depth_maps[fid, cam_idx]).save(mask_file)

            pcd_file = self.out_dir.joinpath(f'{self.out_name}_{self.cnt:06d}_pcd.ply')
            o3d.io.write_point_cloud(str(pcd_file), pcd_combined)
            self.cnt += 1
