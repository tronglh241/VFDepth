import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml
from natsort import natsorted
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', default=str(Path(__file__).parent.joinpath('configs', 'config.yml')))
    args = parser.parse_args()

    # Parse config
    config_file = Path(args.config_file)

    if not config_file.exists():
        raise FileNotFoundError(f'{config_file} not found.')

    with config_file.open() as f:
        config = yaml.safe_load(f)

    out_dir = Path(config['output']['out_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = f'{config["output"]["out_name"]}.mp4'
    fps = config['output']['fps']

    frame_width = config['output']['width']
    frame_height = config['output']['height']

    dir_path = Path(config['input']['dir_path'])
    pattern = config['input']['pattern']

    pcd_files = natsorted(dir_path.glob(pattern))

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=frame_width,
        height=frame_height,
        left=0,
        top=0,
    )

    frame_width, frame_height = np.asarray(vis.capture_screen_float_buffer()).shape[1::-1]

    video_writer = cv2.VideoWriter(
        filename=str(out_dir.joinpath(out_name)),
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=fps,
        frameSize=(frame_width, frame_height),
    )

    render_option = vis.get_render_option()
    render_option.point_size = config['visualization']['point_size']

    with open(config['camera_param_file']) as f:
        view_status = ''.join(list(f))

    vis.set_view_status(view_status)

    for pcd_file in tqdm(pcd_files):
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        pcd = pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(
                np.asarray(config['visualization']['min_bound']),
                np.asarray(config['visualization']['max_bound']),
            )
        )
        vis.clear_geometries()
        vis.add_geometry(pcd, False)
        vis.poll_events()
        vis.update_renderer()

        frame = np.asarray(vis.capture_screen_float_buffer())
        frame = (frame * 255).round().astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        video_writer.write(frame)

    vis.destroy_window()
