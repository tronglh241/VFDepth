from torch import nn

from models.vf_depth.fused_depth_net import FusedDepthNet
from models.vf_depth.fused_pose_net import FusedPoseNet


class VFDepth(nn.Module):
    def __init__(self):
        super(VFDepth, self).__init__()
        self.pose_net = FusedPoseNet()
        self.depth_net = FusedDepthNet()
