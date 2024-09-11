from abc import ABC, abstractmethod

import torch


class Camera(ABC):
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    @abstractmethod
    def project(self, points_3d: torch.Tensor):
        pass

    @abstractmethod
    def inv_project(self, points_2d: torch.Tensor, depth):
        pass

    def inv_project_im(self):
        pass
