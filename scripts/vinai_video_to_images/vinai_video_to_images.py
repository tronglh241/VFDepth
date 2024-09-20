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


def extract_frames(cap, fps=10, neighbor_step=1):
    frame_buffer = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(total_frames):
        # Read current frame
        ret, frame = cap.read()

        if not ret:
            break

        # Check if we need to process the current frame
        if frame_count % fps == 0:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path')
    parser.add_argument('--pattern')
    parser.add_argument('--out-dir', default='output')
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--neighbor-step', type=int, default=1)
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

        files_list_file = _out_dir.joinpath('file_list.txt')

        cap = cv2.VideoCapture(str(video_file))

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

        cnt = 0
        with files_list_file.open(mode='w') as f:
            for frames in tqdm(
                extract_frames(cap, fps=args.fps, neighbor_step=args.neighbor_step),
                leave=False,
                total=n_frames // args.fps,
            ):
                if not all(frame is not None for frame in frames.values()):
                    continue

                for frame_name, frame in frames.items():
                    for cam_name, roi in views.items():
                        im_file = _out_dir.joinpath(f'{cnt:06d}', f'{cam_name}_{frame_name}.jpg')
                        im_file.parent.mkdir(parents=True, exist_ok=True)

                        top = round(frame_height * roi[0][0])
                        bottom = round(frame_height * roi[1][0])
                        left = round(frame_width * roi[0][1])
                        right = round(frame_width * roi[1][1])

                        view = frame[top:bottom, left:right]

                        if args.flip:
                            view = cv2.flip(view, 0)

                        view = cv2.resize(view, (width, height))
                        cv2.imwrite(str(im_file), view)

                f.write(f'{cnt:06d}\n')
                cnt += 1
