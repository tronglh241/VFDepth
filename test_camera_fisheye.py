import cv2
import json
import numpy as np
import torch

from flame_based.models.camera import Fisheye
from fisheye import FisheyeCamera as Fisheye2


def load_calib(cam_data_file, cam_name):
    cam_mapping = {
        'left': 0,
        'front': 1,
        'rear': 2,
        'right': 3,
    }

    with open(cam_data_file) as f:
        cam_data = json.load(f)
        cam_data = {data['camPos']: data for data in cam_data['Items']}

    pos = cam_mapping[cam_name]
    data = cam_data[pos]
    R = np.array(data['matrixR']).reshape(3, 3)
    T = np.array([data['vectT']])
    extrinsic = np.concatenate((R, T.T), axis=1)
    extrinsic = np.concatenate((extrinsic, np.zeros((1, 4))), axis=0)
    extrinsic[-1, -1] = 1.0

    K = np.array(data['matrixK'])
    intrinsic = np.zeros((4, 4))
    intrinsic[0, 0] = K[0]
    intrinsic[0, 2] = K[1]
    intrinsic[1, 1] = K[2]
    intrinsic[1, 2] = K[3]
    intrinsic[2, 2] = 1.0
    intrinsic[3, 3] = 1.0

    distortion = data['matrixD']

    extrinsic = torch.Tensor(extrinsic)
    intrinsic = torch.Tensor(intrinsic)
    distortion = torch.Tensor(distortion)

    return extrinsic, intrinsic, distortion


if __name__ == '__main__':
    cam_data_file = './output/cameraData.json'

    extrinsic, intrinsic, distortion = load_calib(cam_data_file, 'front')

    fisheye = Fisheye(
        width=1280,
        height=800,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        distortion=distortion,
        eps=1e-8,
        max_count=10,
    )
    fisheye2 = Fisheye2('./output/calib_front.json')

    points_3d = torch.Tensor([
        [0, 10, 10],
        [10, 0, 10],
        [10, 10, 0],
    ])

    points_2d, valid, depth = (fisheye.world_to_im(points_3d, normalize=False))
    # print(fisheye2._project(points_3d.numpy()))
    # print(points_2d)
    # print(valid)
    # print(depth)
    # print(fisheye.im_to_cam(points_2d, depth=depth.unsqueeze(-1)))
    undistorted = cv2.fisheye.undistortPoints(
        distorted=points_2d.unsqueeze(1).numpy(),
        K=intrinsic[:3, :3].numpy(),
        D=distortion.numpy(),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )
    print(points_2d)
    print(valid)
    print(depth)
    print(undistorted)
    print(fisheye.im_to_cam(points_2d, depth.unsqueeze(-1)))
    breakpoint()
