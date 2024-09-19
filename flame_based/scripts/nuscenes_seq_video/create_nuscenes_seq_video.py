import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val, train, val
from tqdm import tqdm, trange

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

    # Create NuScenes dataset
    dataset = NuScenes(
        version=config['nuscenes']['version'],
        dataroot=config['nuscenes']['dataroot'],
        verbose=config['nuscenes']['verbose'],
    )

    # Get scenes
    scene_names = config['scenes']

    if len(scene_names) == 1:
        if scene_names[0] == 'train':
            scene_names = train
        elif scene_names[0] == 'val':
            scene_names = val
        elif scene_names[0] == 'mini_train':
            scene_names = mini_train
        elif scene_names[0] == 'mini_val':
            scene_names = mini_val

    scenes = [scene for scene in dataset.scene if scene['name'] in scene_names]

    if len(scene_names) != len(scenes):
        available_scene_names = [scene['name'] for scene in scenes]

        for scene_name in scene_names:
            if scene_name not in available_scene_names:
                print(f'Scene {scene_name} not found.')

        exit(1)

    # Calculate output sizes
    cams = config['cams']
    rows = config['layout']['rows']
    cols = config['layout']['cols']
    input_image_width = config['image_size']['input']['width']
    input_image_height = config['image_size']['input']['height']
    output_image_width = config['image_size']['output']['width']
    output_image_height = config['image_size']['output']['height']
    output_cell_width = round(output_image_width / cols)
    output_cell_height = round(output_image_height / rows)

    fps = config['output']['fps']
    out_dir = Path(config['output']['dirname'])
    out_file = out_dir.joinpath(f'{config["output"]["filename"]}.mp4')
    out_dir.mkdir(parents=True, exist_ok=True)

    video_writer = cv2.VideoWriter(
        str(out_file),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (output_image_width, output_image_height),
    )

    # Define layout
    layout = []

    for row in range(rows):
        layout_row = []
        for col in range(cols):
            index = row * cols + col
            layout_row.append(cams[index] if index < len(cams) else None)
        layout.append(layout_row)

    # For each scene
    for scene in tqdm(scenes, desc='Scene:'):
        nbr_samples = scene['nbr_samples']
        token = scene['first_sample_token']

        # For each sample in the processing scene
        for _ in trange(nbr_samples):
            sample = dataset.get('sample', token)

            # Make sure all sample data are key frames
            for cam in cams:
                sample_data = dataset.get('sample_data', sample['data'][cam])
                assert sample_data['is_key_frame']

            # Read view images and resize to required size
            sample_data_paths = {cam: dataset.get_sample_data_path(sample['data'][cam]) for cam in cams}
            sample_data_imgs = {
                cam: cv2.imread(path)
                for cam, path in sample_data_paths.items()
            }
            sample_data_imgs = {
                cam: cv2.resize(img, (output_cell_width, output_cell_height))
                for cam, img in sample_data_imgs.items()
            }

            # Stack all images according to predefined layout
            layout_imgs = []
            for layout_row in layout:
                layout_row_imgs = [
                    (
                        sample_data_imgs[cam] if cam is not None
                        else np.zeros((output_cell_height, output_cell_width, 3), np.uint8)
                    )
                    for cam in layout_row
                ]
                layout_row_img = np.hstack(layout_row_imgs)
                layout_row_img = cv2.resize(layout_row_img, (output_image_width, output_cell_height))
                layout_imgs.append(layout_row_img)

            layout_img = np.vstack(layout_imgs)
            layout_img = cv2.resize(layout_img, (output_image_width, output_image_height))

            video_writer.write(layout_img)
            token = sample['next']

        # Make sure the last processed sample is the last sample of the processing scene
        assert token == ''

    video_writer.release()
