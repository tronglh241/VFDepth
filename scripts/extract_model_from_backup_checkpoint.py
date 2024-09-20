import argparse
from pathlib import Path

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path')
    parser.add_argument('--model-key', default='model')
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint[args.model_key]

    checkpoint_path = Path(args.checkpoint_path)
    model_path = checkpoint_path.parent.joinpath(f'model_{checkpoint_path.name}')

    torch.save(model_state_dict, str(model_path))
