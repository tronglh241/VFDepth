import json
import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

sys.path.append(os.getcwd())
from models.camera import Fisheye  # noqa: E402


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
    calib_file = 'tests/cameraData.json'
    calib_info = load_calib(calib_file)
    for cam_name in ['front', 'left', 'rear', 'right']:
        # cam_name = 'front'
        intrinsic = torch.eye(4)
        intrinsic[:3, :3] = calib_info[cam_name]['intrinsic']
        fisheye = Fisheye(
            width=1280,
            height=800,
            intrinsic=intrinsic,
            extrinsic=calib_info[cam_name]['extrinsic'],
            distortion=calib_info[cam_name]['distortion'],
        )

        src_im_file = (
            'data/PHOTOMETRIC_ALIGNMENT_FISHEYE_VIDEO_calib17Feb2023/'
            '20_PA- 3D view at 4 overlap zones When passing throught shade of building or tree/'
            'PA- 3D view at 4 overlap zones When passing throught shade of building or tree_Sunny/'
            f'000010/{cam_name}_current.jpg'
        )
        src_mask_file = f'datasets/masks/vinaicar/{cam_name}.png'
        src_im = cv2.imread(src_im_file)
        src_mask = cv2.imread(src_mask_file, cv2.IMREAD_GRAYSCALE)
        height, width = src_mask.shape

        src_im = torch.from_numpy(src_im).unsqueeze(0)
        src_mask = torch.from_numpy(src_mask).unsqueeze(0).unsqueeze(0)
        dst_depth = torch.full((1, height, width, 1), 10)

        points_3d, valid_points_3d = fisheye.im_to_cam_map(dst_depth, src_mask)
        points_3d = points_3d[valid_points_3d]

        points_3d = fisheye.cam_to_world(points_3d)
        points_2d, valid_points_2d, depth = fisheye.world_to_im(points_3d, False)

        image = torch.zeros((3, 800, 1280))
        points_2d = points_2d.round().int()[valid_points_2d]
        image[:, points_2d[:, 1], points_2d[:, 0]] = 1
        transforms.ToPILImage()(image).save(f'{cam_name}.png')

    # transforms.ToPILImage()(valid_points_3d.reshape(height, width).float()).save('valid.png')
    # width = 100
    # height = 150
    # y, z = torch.meshgrid(
    #     torch.linspace(-width, width, steps=100),
    #     torch.linspace(-height, height, steps=100),
    #     indexing='xy',
    # )
    # x = 10
    # points_3d = torch.stack([
    #     torch.full(y.shape, x),
    #     y,
    #     z
    # ], dim=-1).view(-1, 3)

    # points_2d, valid_points, depth = fisheye.world_to_im(points_3d, normalize=False)
    # image = torch.zeros((3, 800, 1280))
    # points_2d = points_2d.round().int()[valid_points]
    # image[:, points_2d[:, 1], points_2d[:, 0]] = 1
    # transforms.ToPILImage()(image).save('a.png')
    # transforms.ToPILImage()(valid_points.reshape(100, 100).float()).save('valid.png')

    # src_img_file =
