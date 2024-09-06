from torch import nn

from models.vf_depth.fusion_pose_net import FusedPoseNet


class VFDepth(nn.Module):
    def __init__(self, arg):
        super(VFDepth, self).__init__()
        self.pose_net = FusedPoseNet()
        self.depth_net = arg
