import argparse
from pathlib import Path

import cv2
from natsort import natsorted
from tqdm import tqdm

views = {
    'rear': (
        (0.0, 0.0),
        (0.5, 0.5)
    ),
    'right': (
        (0.0, 0.5),
        (0.5, 1.0),
    ),
    'left': (
        (0.5, 0.0),
        (1.0, 0.5),
    ),
    'front': (
        (0.5, 0.5),
        (1.0, 1.0),
    ),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('--pattern')
    parser.add_argument('--out-dir', default='output')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--flip', action='store_true')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)

    if args.pattern:
        video_files = natsorted(data_path.glob(args.pattern), key=lambda x: x.stem)
    else:
        video_files = [data_path]

    for video_file in tqdm(video_files):
        _out_dir = out_dir.joinpath(video_file.stem)
        _out_dir.mkdir(parents=True, exist_ok=True)

        files_list_file = _out_dir.joinpath(f'{video_file.stem}.txt')

        with files_list_file.open(mode='w') as f:
            cap = cv2.VideoCapture(str(video_file))

            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if args.width and args.height:
                width = args.width
                height = args.height
            else:
                width = frame_width
                height = frame_height

            cam_out_dirs = {
                cam_name: _out_dir.joinpath(f'{cam_name}')
                for cam_name in views
            }

            for cam_out_dir in cam_out_dirs.values():
                cam_out_dir.mkdir(parents=True, exist_ok=True)

            for frame_id in tqdm(range(n_frames), leave=False):
                _, frame = cap.read()

                for cam_name, roi in views.items():
                    cam_out_dir = cam_out_dirs[cam_name]

                    top = round(frame_height * roi[0][0])
                    bottom = round(frame_height * roi[1][0])
                    left = round(frame_width * roi[0][1])
                    right = round(frame_width * roi[1][1])

                    view = frame[top:bottom, left:right]

                    if args.flip:
                        view = cv2.flip(view, 0)

                    view = cv2.resize(view, (width, height))
                    im_file = cam_out_dir.joinpath(f'{frame_id:06d}.jpg')
                    cv2.imwrite(str(im_file), view)

                f.write(f'{im_file.stem}\n')
