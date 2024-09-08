import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(os.getcwd())
import utils  # noqa: E402
from models import VFDepthAlgo  # noqa: E402
from models.vf_depth.vf_depth import VFDepth  # noqa: E402

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

    cfg = utils.get_config(config['org_model_config'], mode='train')
    org_model = VFDepthAlgo(cfg, 0)
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
