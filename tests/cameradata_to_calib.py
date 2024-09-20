import argparse
import json
from pathlib import Path

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cam_data_file')
    parser.add_argument('--out-dir', default='output')
    args = parser.parse_args()

    cam_data_file = Path(args.cam_data_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_mapping = {
        'left': 0,
        'front': 1,
        'rear': 2,
        'right': 3,
    }

    with cam_data_file.open(mode='r') as f:
        cam_data = json.load(f)
        cam_data = {data['camPos']: data for data in cam_data['Items']}

    for cam_name, pos in cam_mapping.items():
        data = cam_data[pos]
        R = np.array(data['matrixR']).reshape(3, 3)
        T = np.array([data['vectT']])
        extrinsic = np.concatenate((R, T.T), axis=1)

        K = np.array(data['matrixK'])
        intrinsic = np.zeros((3, 3))
        intrinsic[0, 0] = K[0]
        intrinsic[0, 2] = K[1]
        intrinsic[1, 1] = K[2]
        intrinsic[1, 2] = K[3]
        intrinsic[2, 2] = 1.0

        distortion = data['matrixD']

        with out_dir.joinpath(f'calib_{cam_name}.json').open(mode='w') as f:
            calib = {
                'extrinsic': extrinsic.tolist(),
                'intrinsic': intrinsic.tolist(),
                'distortion': distortion,
            }
            json.dump(calib, f, indent=4)
