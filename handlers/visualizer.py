from pathlib import Path
from typing import Any, Callable, Tuple

import cv2
import torch
import torch.nn.functional as F
from flame.handlers import Handler
from ignite.engine import Events
from ignite.handlers import global_step_from_engine
from matplotlib import cm
from torchvision.utils import make_grid


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
        event_name: Any = Events.ITERATION_COMPLETED,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_name = out_name
        self.out_resolution = out_resolution
        self.fps = fps
        self.fourcc = fourcc
        self.file_ext = file_ext
        self.out_writer = None

        self.grid_nrow = grid_nrow
        self.output_transform = output_transform
        self.cmap = torch.Tensor(cm.get_cmap('plasma').colors)

        self.global_step_from_engine = global_step_from_engine(trainer, event_name)

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
        count = self.global_step_from_engine('', '')
        out_name = f'{self.out_name}_{count}.{self.file_ext}'
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
        depth_maps = torch.matmul(
            F.one_hot(torch.round(depth_maps * 255.0).to(torch.int64), num_classes=256).to(torch.float32),
            self.cmap.to(depth_maps.device),
        )
        depth_maps = depth_maps.permute(0, 1, 2, 5, 3, 4).squeeze(2)
        assert len(depth_maps.shape) == 5

        for depth_map, image in zip(depth_maps, images):
            vis_image = torch.cat([depth_map, image], dim=2)
            vis_image = make_grid(vis_image, nrow=self.grid_nrow)

            if vis_image.shape[-2:] != (self.out_resolution[1], self.out_resolution[0]):
                vis_image = F.interpolate(
                    vis_image.unsqueeze(0),
                    (self.out_resolution[1], self.out_resolution[0]),
                ).squeeze(0)

            vis_image = vis_image[[2, 1, 0]] * 255.0  # RGB to BGR
            vis_image = vis_image.round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()

            self.out_writer.write(vis_image)
