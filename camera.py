import json
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np


class Calib:
    def __init__(
        self,
        file: str,
    ):
        self.file = file
        self._load()

    def _load(self) -> None:
        with open(self.file, mode='r') as f:
            info = json.load(f)

        self.extrinsic = np.array(info['extrinsic'])
        self.intrinsic = np.array(info['intrinsic'])
        self.distortion = np.array(info['distortion']) if 'distortion' in info else None

        if self.extrinsic.shape != (3, 4):
            raise Exception(f'Extrinsic in file {self.file} must have shape of (3, 4).')

        if self.intrinsic.shape != (3, 3):
            raise Exception(f'Intrinsic in file {self.file} must have shape of (3, 3).')

        if self.distortion is not None and self.distortion.shape != (4,):
            raise Exception(f'Distortion in file {self.file} must have shape of (4,).')


class Camera(ABC):
    def __init__(
        self,
        calib_file: str,
        fov: Optional[Union[float, Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
    ):
        self.calib = Calib(calib_file)

        if fov is None:
            self.horizontal_fov = self.vertical_fov = (-np.pi / 2, np.pi / 2)
        elif isinstance(fov, tuple):
            if isinstance(fov[0], tuple):
                self.horizontal_fov = fov[0]
                self.vertical_fov = fov[1]
            else:
                self.horizontal_fov = (-fov[0] / 2, fov[0] / 2)
                self.vertical_fov = (-fov[1] / 2, fov[1] / 2)
        else:
            self.horizontal_fov = (-fov / 2, fov / 2)
            self.vertical_fov = (-fov / 2, fov / 2)

    @abstractmethod
    def map(
        self,
        points: np.ndarray,
        im_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @property
    def center(self):
        extrinsic: np.ndarray = np.concatenate((self.calib.extrinsic, np.array([[0, 0, 0, 1]])))
        center = np.array([[0, 0, 0, 1]])
        center = np.linalg.inv(extrinsic) @ center.T
        center = center.T[0]
        return center
