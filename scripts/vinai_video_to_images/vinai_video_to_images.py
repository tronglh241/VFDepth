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


def extract_frames(frame_iter, frame_step=10, neighbor_step=1):
    frame_buffer = []
    frame_count = 0

    for frame in frame_iter:
        # Check if we need to process the current frame
        if frame_count % frame_step == 0:
            # Add current frame to buffer
            frame_buffer.append((frame_count, frame))

            # Remove old frames if buffer gets too big
            if len(frame_buffer) > 2 * neighbor_step + 1:
                frame_buffer.pop(0)

            # Return current frame
            current_frame = frame_buffer[len(frame_buffer) // 2][1]
            prev_frame_idx = len(frame_buffer) // 2 - neighbor_step
            next_frame_idx = len(frame_buffer) // 2 + neighbor_step

            prev_frame = frame_buffer[prev_frame_idx][1] if prev_frame_idx >= 0 else None
            next_frame = frame_buffer[next_frame_idx][1] if next_frame_idx < len(frame_buffer) else None

            yield {
                'current': current_frame,
                'prev': prev_frame,
                'next': next_frame
            }

        frame_count += 1


def split_view(frame):
    frame_by_cam = {}
    frame_width, frame_height = frame.shape[1::-1]

    for cam_name, roi in views.items():
        top = round(frame_height * roi[0][0])
        bottom = round(frame_height * roi[1][0])
        left = round(frame_width * roi[0][1])
        right = round(frame_width * roi[1][1])

        view = frame[top:bottom, left:right]

        if args.flip:
            view = cv2.flip(view, 0)

        frame_by_cam[cam_name] = view

    return frame_by_cam


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('--pattern')
    parser.add_argument('--out-dir', default='output')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--frame-step', type=int, default=10)
    parser.add_argument('--neighbor-step', type=int, default=1)
    parser.add_argument('--flip', action='store_true')
    args = parser.parse_args()

    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)

    if args.pattern:
        data_paths = natsorted(data_path.glob(args.pattern), key=lambda x: x.stem)
    else:
        data_paths = [data_path]

    for path in tqdm(data_paths):
        _out_dir = out_dir.joinpath(path.stem if path.is_file() else path.name)
        _out_dir.mkdir(parents=True, exist_ok=True)

        files_list_file = _out_dir.joinpath('file_list.txt')

        cam_out_dirs = {
            cam_name: _out_dir.joinpath(f'{cam_name}')
            for cam_name in views
        }

        if path.is_file():  # video
            cap = cv2.VideoCapture(str(path))

            n_frames = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if args.width and args.height:
                width = args.width
                height = args.height
            else:
                width = frame_width
                height = frame_height

            frames_by_view = (split_view(cap.read()[1]) for _ in range(n_frames))
        else:  # folder
            img_files_by_view = {cam_name: natsorted(path.glob(f'*{cam_name}*')) for cam_name in views}
            frame_width, frame_height = cv2.imread(img_files_by_view[list(views.keys())[0]][0]).shape[1::-1]

            if args.width and args.height:
                width = args.width
                height = args.height
            else:
                width = frame_width
                height = frame_height
            n_frames = len(list(img_files_by_view.values())[0])
            frames_by_view = (
                {key: cv2.imread(img_files_by_view[key][i]) for key in views.keys()}
                for i in range(n_frames)
            )
        cnt = 0
        with files_list_file.open(mode='w') as f:
            for frames in tqdm(
                extract_frames(frames_by_view, frame_step=args.frame_step, neighbor_step=args.neighbor_step),
                leave=False,
                total=n_frames // args.frame_step,
            ):
                if not all(frame is not None for frame in frames.values()):
                    continue

                for frame_name, frame in frames.items():
                    for cam_name, view in frame.items():
                        view = cv2.resize(view, (width, height))
                        im_file = _out_dir.joinpath(f'{cnt:06d}', f'{cam_name}_{frame_name}.jpg')
                        im_file.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(im_file), view)

                f.write(f'{cnt:06d}\n')
                cnt += 1
