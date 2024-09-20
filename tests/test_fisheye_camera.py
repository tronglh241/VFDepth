import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

sys.path.append(os.getcwd())
from models.camera import Fisheye  # noqa: E402


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


class FisheyeCamera(Camera):
    def map(
        self,
        points: np.ndarray,
        im_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        points_2d = self._project(points)
        cam_points = self._world_to_cam(points)
        points_in_hor_fov = self._in_fov(cam_points[:, 0], cam_points[:, 2], self.horizontal_fov)
        points_in_ver_fov = self._in_fov(cam_points[:, 1], cam_points[:, 2], self.vertical_fov)
        points_in_image = self._in_image(points_2d, im_size)
        return points_2d, np.logical_and.reduce([points_in_hor_fov, points_in_ver_fov, points_in_image])

    def _world_to_cam(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        if len(points.shape) != 2 or points.shape[1] != 3:
            raise Exception(f'`points` must have shape of (n, 3), found {points.shape}.')

        points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
        points = self.calib.extrinsic @ points.T
        points = points.T

        return points

    def _in_fov(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        fov: Tuple[float, float],
    ) -> np.ndarray:
        return np.logical_and(fov[0] <= np.arctan2(x1, x2), np.arctan2(x1, x2) <= fov[1])

    def _project(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        rmat = self.calib.extrinsic[:, :3]
        rvec = cv2.Rodrigues(rmat)[0].flatten()
        tvec = self.calib.extrinsic[:, 3:].flatten()

        rvec = np.array([[rvec]])
        tvec = np.array([[tvec]])

        points, _ = cv2.fisheye.projectPoints(
            objectPoints=np.expand_dims(points, axis=0),
            rvec=rvec,
            tvec=tvec,
            K=self.calib.intrinsic,
            D=self.calib.distortion,
        )
        points = points[0]
        return points

    def _in_image(
        self,
        points: np.ndarray,
        im_size: Tuple[int, int],
    ) -> np.ndarray:
        in_width = np.logical_and(points[:, 0] >= 0, points[:, 0] < im_size[0])
        in_height = np.logical_and(points[:, 1] >= 0, points[:, 1] < im_size[1])
        return np.logical_and(in_width, in_height)


def load_calib(cam_data_file):
    cam_mapping = {
        'left': 0,
        'front': 1,
        'rear': 2,
        'right': 3,
    }

    with open(cam_data_file) as f:
        cam_data = json.load(f)
        cam_data = {data['camPos']: data for data in cam_data['Items']}

    calib_info = {}

    for cam_name in cam_mapping:
        pos = cam_mapping[cam_name]
        data = cam_data[pos]
        R = np.array(data['matrixR']).reshape(3, 3)
        T = np.array([data['vectT']])
        extrinsic = np.concatenate((R, T.T), axis=1)
        extrinsic = np.concatenate((extrinsic, np.zeros((1, 4))), axis=0)
        extrinsic[-1, -1] = 1.0

        K = np.array(data['matrixK'])
        intrinsic = np.zeros((3, 3))
        intrinsic[0, 0] = K[0]
        intrinsic[0, 2] = K[1]
        intrinsic[1, 1] = K[2]
        intrinsic[1, 2] = K[3]
        intrinsic[2, 2] = 1.0

        distortion = data['matrixD']

        extrinsic = torch.Tensor(extrinsic)
        intrinsic = torch.Tensor(intrinsic)
        distortion = torch.Tensor(distortion)

        calib_info[cam_name] = {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'distortion': distortion,
        }

    return calib_info


if __name__ == '__main__':
    calib_file = 'data/cameraData.json'
    calib_info = load_calib(calib_file)
    intrinsic = torch.eye(4)
    intrinsic[:3, :3] = calib_info['front']['intrinsic']
    fisheye = Fisheye(
        width=1280,
        height=800,
        intrinsic=intrinsic,
        extrinsic=calib_info['front']['extrinsic'],
        distortion=calib_info['front']['distortion'],
        eps=1e-8,
        max_count=10,
    )

    fisheye2 = FisheyeCamera('tests/calib_front.json')

    # points_3d = [
    #     [0, 0, 0],
    #     [10, 0, 0],
    #     [0, 10, 0],
    #     [0, 0, 10],
    #     [10, 10, 10],
    # ]
    points_3d = [[torch.randint(50, (1,)).item() for _ in range(3)] for _ in range(5)]

    points_2d, valid, depth = fisheye.world_to_im(torch.Tensor(points_3d), False)
    points_2d_2 = fisheye2._project(np.asarray(points_3d).astype(np.float32))
    print(points_2d.numpy() - points_2d_2)

    undistorted = fisheye.im_to_cam(torch.from_numpy(points_2d_2), torch.ones((len(points_3d), 1)))
    undistorted2 = cv2.fisheye.undistortPoints(
        np.asarray([points_2d_2]),
        K=fisheye2.calib.intrinsic,
        D=fisheye2.calib.distortion,
    )
    print(undistorted[valid].squeeze(1)[:, :2].numpy() - undistorted2[0][valid.numpy()])
