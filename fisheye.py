from typing import Tuple

import cv2
import numpy as np

from camera import Camera


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
