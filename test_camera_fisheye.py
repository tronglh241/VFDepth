import torch
from flame_based.models.camera import Fisheye

import utils
from models import VFDepthAlgo

if __name__ == '__main__':
    cfg = utils.get_config('./configs/nuscenes/nusc_surround_fusion.yaml', mode='train')

    org_model = VFDepthAlgo(cfg, 0)

    dataset = org_model.train_dataloader()
    inputs = next(iter(dataset))

    fisheye = Fisheye(
        width=640,
        height=352,
        extrinsic=inputs['extrinsics_inv'],
        intrinsic=inputs['intrinsic'],
        distortion: torch.Tensor,
        eps: float=1e-8,
        max_count: int=10)
