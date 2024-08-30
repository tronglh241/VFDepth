import time

import torch
import utils
from calflops import calculate_flops
from models.vfdepth import VFDepthAlgo
from tqdm import trange

config_file = 'configs/nuscenes/nusc_surround_fusion.yaml'

cfg = utils.get_config(config_file, mode='train')
model = VFDepthAlgo(cfg, 0)

inputs = {
    'extrinsics': torch.rand([1, 6, 4, 4]).cuda(),
    'mask': torch.rand([1, 6, 1, 352, 640]).cuda(),
    ('K', 0): torch.rand([1, 6, 4, 4]).cuda(),
    ('inv_K', 0): torch.rand([1, 6, 4, 4]).cuda(),
    ('color', 0, 0): torch.rand([1, 6, 3, 352, 640]).cuda(),
    ('color_aug', 0, 0): torch.rand([1, 6, 3, 352, 640]).cuda(),
    ('K', 1): torch.rand([1, 6, 4, 4]).cuda(),
    ('inv_K', 1): torch.rand([1, 6, 4, 4]).cuda(),
    ('color', 0, 1): torch.rand([1, 6, 3, 176, 320]).cuda(),
    ('color_aug', 0, 1): torch.rand([1, 6, 3, 176, 320]).cuda(),
    ('K', 2): torch.rand([1, 6, 4, 4]).cuda(),
    ('inv_K', 2): torch.rand([1, 6, 4, 4]).cuda(),
    ('color', 0, 2): torch.rand([1, 6, 3, 88, 160]).cuda(),
    ('color_aug', 0, 2): torch.rand([1, 6, 3, 88, 160]).cuda(),
    ('K', 3): torch.rand([1, 6, 4, 4]).cuda(),
    ('inv_K', 3): torch.rand([1, 6, 4, 4]).cuda(),
    ('color', 0, 3): torch.rand([1, 6, 3, 44, 80]).cuda(),
    ('color_aug', 0, 3): torch.rand([1, 6, 3, 44, 80]).cuda(),
    ('color', -1, 0): torch.rand([1, 6, 3, 352, 640]).cuda(),
    ('color_aug', -1, 0): torch.rand([1, 6, 3, 352, 640]).cuda(),
    ('color', 1, 0): torch.rand([1, 6, 3, 352, 640]).cuda(),
    ('color_aug', 1, 0): torch.rand([1, 6, 3, 352, 640]).cuda()
}
model.set_val()
with torch.no_grad():
    output = model.process_batch(inputs, 0)

    pose_net = model.models['pose_net']
    pose_net(inputs, [0, 1], None)

    depth_net = model.models['depth_net']
    depth_net(inputs)

flops, macs, params = calculate_flops(
    model=pose_net,
    args=(inputs, [0, 1], None),
    print_detailed=False,
    print_results=False,
    output_as_string=True,
    output_precision=4)
print("Pose Net FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

flops, macs, params = calculate_flops(
    model=depth_net,
    args=(inputs,),
    output_as_string=True,
    print_detailed=False,
    print_results=False,
    output_precision=4)
print("Depth Net FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

with torch.no_grad():
    total = 0.
    n = 300
    for _ in trange(n):
        start = time.time()
        output = pose_net(inputs, [0, 1], None)
        stop = time.time()
        total += stop - start

    print(f'PoseNet Time per frame: {total / n}s')

with torch.no_grad():
    total = 0.
    n = 300
    for _ in trange(n):
        start = time.time()
        output = depth_net(inputs)
        stop = time.time()
        total += stop - start

    print(f'Depth Time per frame: {total / n}s')
