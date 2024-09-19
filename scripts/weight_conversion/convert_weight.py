import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(os.getcwd())
from models.vf_depth import VFDepth  # noqa: E402

sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth')))
sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth', 'external', 'packnet_sfm')))
sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth', 'external', 'dgp')))
sys.path.append(str(Path(os.getcwd()).joinpath('VFDepth', 'external', 'monodepth2')))
from VFDepth import utils  # noqa: E402
from VFDepth.models import VFDepthAlgo  # noqa: E402


class VFDepthAlgoModified(VFDepthAlgo):
    def __init__(
        self,
        config_file,
        log_dir,
        weights,
        models_to_load,
    ):
        cfg = utils.get_config(config_file, mode='train')

        log_path = os.path.join(log_dir, os.path.splitext(os.path.basename(config_file))[0])
        weight_path = os.path.join(log_path, 'models', weights)
        cfg['data']['load_weights_dir'] = weight_path

        self.read_config(cfg)
        self.models_to_load = models_to_load
        self.models = self.prepare_model(cfg, 0)
        self.set_optimizer()


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

    org_model = VFDepthAlgoModified(
        config_file=config['org']['config_file'],
        log_dir=config['org']['log_dir'],
        weights=config['org']['weights'],
        models_to_load=config['org']['models_to_load'],
    )
    org_model.load_weights()
    model = VFDepth().cuda()

    model.pose_net.encoder.load_state_dict(org_model.models['pose_net'].encoder.state_dict())
    model.pose_net.conv1x1.load_state_dict(org_model.models['pose_net'].conv1x1.state_dict())
    model.pose_net.fusion_net.load_state_dict(org_model.models['pose_net'].fusion_net.state_dict())
    model.pose_net.decoder.load_state_dict(org_model.models['pose_net'].pose_decoder.state_dict())

    model.depth_net.encoder.load_state_dict(org_model.models['depth_net'].encoder.state_dict())
    model.depth_net.conv1x1.load_state_dict(org_model.models['depth_net'].conv1x1.state_dict())
    model.depth_net.fusion_net.load_state_dict(org_model.models['depth_net'].fusion_net.state_dict())
    model.depth_net.decoder.load_state_dict(org_model.models['depth_net'].decoder.state_dict())

    out_file = Path(config['output']['dirname']).joinpath(f'{config["output"]["filename"]}.pt')
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model.cpu()
    torch.save(model.state_dict(), out_file)
