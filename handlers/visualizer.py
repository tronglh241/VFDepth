import shutil
from pathlib import Path

import numpy as np
import PIL.Image as pil
from flame.engine import Evaluator, Trainer
from flame.handlers import Handler
from ignite.engine import Events
from utils.visualize import colormap
from collections import defaultdict


class Visualizer(Handler):
    def __init__(self, trainer: Trainer, evaluator: Evaluator, out_dir: str):
        self.out_dir = Path(out_dir)

        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)

        self.out_dir.mkdir(parents=True)

        self.trainer = trainer
        self.cnt = defaultdict(int)
        actions = []
        action = {
            'engine': evaluator,
            'event': Events.ITERATION_COMPLETED,
            'func': self,
        }
        actions.append(action)
        action = {
            'engine': evaluator,
            'event': Events.STARTED,
            'func': self.reset,
        }
        actions.append(action)
        super(Visualizer, self).__init__(actions=actions)

    def reset(self):
        self.cnt = defaultdict(int)

    def __call__(self, engine):
        y_pred, y = engine.state.output
        epoch = self.trainer.state.epoch

        out_dir = self.out_dir.joinpath(str(epoch))
        outputs, _ = y_pred
        scale = 0
        for key in outputs:
            if isinstance(key, tuple) and key[0] == 'cam':
                cam_id = key[1]
                target_view = outputs[('cam', cam_id)]
                disps = target_view['disp', scale]

                cam_dir = out_dir.joinpath(f'cam{cam_id}')
                cam_dir.mkdir(parents=True, exist_ok=True)

                for jdx, disp in enumerate(disps):
                    disp = colormap(disp)[0, ...].transpose(1, 2, 0)
                    disp = pil.fromarray((disp * 255).astype(np.uint8))
                    cur_idx = self.cnt[cam_id]
                    disp.save(cam_dir.joinpath(f'{cur_idx:03d}_disp.jpg'))
                    self.cnt[cam_id] += 1
