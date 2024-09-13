import math
from pathlib import Path
from typing import Callable, Tuple

import cv2
import numpy as np
from flame.handlers import Handler
from ignite.engine import Events
from matplotlib import cm


class Visualizer(Handler):
    def __init__(
        self,
        trainer,
        evaluator,
        out_dir: str,
        out_name: str,
        out_resolution: Tuple[int, int],
        fps: int,
        file_ext: str = 'mp4',
        fourcc: str = 'mp4v',
        grid_nrow: int = 3,
        output_transform: Callable = lambda x: x,
    ):
        self.trainer = trainer
        self.evaluator = evaluator

        self.out_dir = Path(out_dir)
        self.out_dir.parent.mkdir(parents=True, exist_ok=True)
        self.out_name = out_name
        self.out_resolution = out_resolution
        self.fps = fps
        self.fourcc = fourcc
        self.file_ext = file_ext
        self.out_writer = None

        self.grid_nrow = grid_nrow
        self.output_transform = output_transform
        self.cmap = cm.get_cmap('plasma')

        actions = []

        action = {
            'engine': evaluator,
            'event': Events.EPOCH_STARTED,
            'func': self.create_writer,
        }
        actions.append(action)

        action = {
            'engine': evaluator,
            'event': Events.EPOCH_COMPLETED,
            'func': self.release_writer,
        }
        actions.append(action)

        action = {
            'engine': evaluator,
            'event': Events.ITERATION_COMPLETED,
            'func': self.save_depth_map,
        }
        actions.append(action)

        super(Visualizer, self).__init__(actions=actions)

    def create_writer(self, engine):
        epoch = self.trainer.state.epoch
        out_name = f'{self.out_name}_{epoch}.{self.file_ext}'
        out_file = str(self.out_dir.joinpath(out_name))

        self.out_writer = cv2.VideoWriter(
            out_file,
            cv2.VideoWriter_fourcc(*self.fourcc),
            self.fps,
            self.out_resolution,
        )

        if not self.out_writer.isOpened():
            raise RuntimeError(f'Unsupported codec {self.fourcc}')

    def release_writer(self):
        self.out_writer.release()

    def save_depth_map(self, engine):
        depth_maps, images = self.output_transform(engine.state.output)

        for depth_map, image in zip(depth_maps, images):
            cam_vis_images = []
            for cam_depth_map, cam_image in zip(depth_map, image):
                cam_depth_map = cam_depth_map.detach().cpu().numpy()
                assert cam_depth_map.shape[0] == 1

                color_depth_map = self.cmap(cam_depth_map[0])[:, :, :3].astype(np.float32)
                cam_image = cam_image.permute(1, 2, 0).cpu().numpy().astype(np.float32)

                cam_vis_image = cv2.vconcat([color_depth_map, cam_image])
                cam_vis_images.append(cam_vis_image)

            num_rows = math.ceil(len(cam_vis_images) / self.grid_nrow)
            num_cols = self.grid_nrow
            grid_cell_width = self.out_resolution[0] // num_cols
            grid_cell_height = self.out_resolution[0] // num_rows

            cam_vis_images = [cv2.resize(im, (grid_cell_width, grid_cell_height)) for im in cam_vis_images]

            vis_images = []
            for i in range(num_rows):
                row_images = []

                for j in range(num_cols):
                    if i * num_cols + j < len(cam_vis_images):
                        row_images.append(cam_vis_images[i * num_cols + j])
                    else:
                        row_images.append(np.zeros((grid_cell_height, grid_cell_width, 3), dtype=float))

                row_image = cv2.hconcat(row_images)
                vis_images.append(row_image)

            vis_image = cv2.vconcat(vis_images)
            vis_image = cv2.resize(vis_image, self.out_resolution)
            vis_image = (vis_image * 255).astype(np.uint8)
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

            self.out_writer.write(vis_image)
