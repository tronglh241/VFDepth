from torch import nn
from external.layers import ResnetEncoder
from network.blocks import conv2d
from models.vf_depth.vfnet import VFNet
from collections import OrderedDict


class DepthDecoder(nn.Module):
    """
    This class decodes encoded 2D features to estimate depth map.
    Unlike monodepth depth decoder, we decode features with corresponding level
    we used to project features in 3D (default: level 2(H/4, W/4))
    """

    def __init__(
        self,
        level_in,
        num_ch_enc,
        num_ch_dec,
        use_skips=False,
    ):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = 1
        self.use_skips = use_skips

        self.level_in = level_in
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = num_ch_dec

        self.convs = OrderedDict()
        for i in range(self.level_in, -1, -1):
            num_ch_in = self.num_ch_enc[-1] if i == self.level_in else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 0)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin='ELU')

            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[('upconv', i, 1)] = conv2d(num_ch_in, num_ch_out, kernel_size=3, nonlin='ELU')

        for s in self.scales:
            self.convs[('dispconv', s)] = conv2d(self.num_ch_dec[s], self.num_output_channels, 3, nonlin = None)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        outputs = {}

        # decode
        x = input_features[-1]
        for i in range(self.level_in, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
            if i in self.scales:
                outputs[('disp', i)] = self.sigmoid(self.convs[('dispconv', i)](x))
        return outputs


class FusedDepthNet(nn.Module):
    def __init__(
        self,
        resnet_num_layers: int = 18,
        resnet_pretrained: bool = True,
        fusion_level: int = 2,  # zero-based level, Resnet has 5-layer, e.g, 2 means 3rd layer.
        fusion_feat_in_dim: int = 256,  # number of channels of fused feature map for each input image
    ):
        super(FusedDepthNet, self).__init__()
        # feature encoder
        # resnet feat: 64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)
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
        self.fusion_net = VFNet(fusion_feat_in_dim, fusion_feat_out_dim, model='depth')

        # depth decoder
        num_ch_enc = self.encoder.num_ch_enc[:(fusion_level + 1)]
        num_ch_dec = [16, 32, 64, 128, 256]
        self.decoder = DepthDecoder(fusion_level, num_ch_enc, num_ch_dec, use_skips=self.use_skips)
