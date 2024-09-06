from external.layers import PoseDecoder, ResnetEncoder
from models.vf_depth.vfnet import VFNet
from network.blocks import conv2d
from torch import nn


class FusedPoseNet(nn.Module):
    def __init__(
        self,
        resnet_num_layers: int = 18,
        resnet_pretrained: bool = True,
        fusion_level: int = 2,  # zero-based level, Resnet has 5-layer, e.g, 2 means 3rd layer.
        fusion_feat_in_dim: int = 256,  # number of channels of fused feature map for each input image
    ):
        super(FusedPoseNet, self).__init__()
        self.encoder = ResnetEncoder(
            num_layers=resnet_num_layers,
            pretrained=resnet_pretrained,
            num_input_images=2,  # Pose estimation requires two consecutive frames
        )

        # Only feature extractor needed, fully connected layer is not required
        del self.encoder.encoder.fc

        # 1x1 convolution for channel reduction in multi-layer feature map fusion
        enc_feat_dim = sum(self.encoder.num_ch_enc[fusion_level:])
        self.conv1x1 = conv2d(enc_feat_dim, fusion_feat_in_dim, kernel_size=1, padding_mode='reflect')

        # fusion net
        fusion_feat_out_dim = self.encoder.num_ch_enc[fusion_level]
        self.fusion_net = VFNet(fusion_feat_in_dim, fusion_feat_out_dim, model='pose')

        # depth decoder
        self.pose_decoder = PoseDecoder(
            num_ch_enc=[fusion_feat_out_dim],
            num_input_features=1,
            num_frames_to_predict_for=1,
            stride=2,
        )
