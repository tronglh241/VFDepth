import argparse
from pathlib import Path

import yaml
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val, train, val
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

    out_dir = Path(config['output']['dirname'])
    out_file = out_dir.joinpath(f'{config["output"]["filename"]}.txt')
    out_dir.mkdir(parents=True, exist_ok=True)

    with out_file.open(mode='w') as f:
        # For each scene
        for scene in tqdm(scenes, desc='Scene:'):
            nbr_samples = scene['nbr_samples']
            first_sample = dataset.get('sample', scene['first_sample_token'])
            token = first_sample['next']

            seq_length = 0
            while token != scene['last_sample_token']:
                f.write(f'{token}\n')
                sample = dataset.get('sample', token)
                token = sample['next']
                seq_length += 1

            # Make sure the last processed sample is the last sample of the processing scene
            assert token == scene['last_sample_token']
            assert seq_length == nbr_samples - 2
