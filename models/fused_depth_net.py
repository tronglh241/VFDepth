from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import conv2d, pack_cam_feat, unpack_cam_feat, upsample
from .resnet_encoder import ResnetEncoder
from .vf_net import DepthVFNet


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

        self.convs[('dispconv', 0)] = conv2d(self.num_ch_dec[0], self.num_output_channels, 3, nonlin=None)
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        # decode
        x = input_features[-1]
        for i in range(self.level_in, -1, -1):
            x = self.convs[('upconv', i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[('upconv', i, 1)](x)
        depth_map = self.sigmoid(self.convs[('dispconv', 0)](x))
        return depth_map


class FusedDepthNet(nn.Module):
    def __init__(
        self,
        resnet_num_layers: int = 18,
        resnet_pretrained: bool = True,
        fusion_level: int = 2,  # zero-based level, Resnet has 5-layer, e.g, 2 means 3rd layer.
        fusion_feat_in_dim: int = 256,  # number of channels of fused feature map for each input image
        input_width: int = 640,
        input_height: int = 352,
        use_skips: bool = False,
    ):
        super(FusedDepthNet, self).__init__()
        # feature encoder
        # resnet feat: 64(1/2), 64(1/4), 128(1/8), 256(1/16), 512(1/32)
        self.encoder = ResnetEncoder(
            num_layers=resnet_num_layers,
            pretrained=resnet_pretrained,
            num_input_images=1,  # Pose estimation requires two consecutive frames
        )

        # Only feature extractor needed, fully connected layer is not required
        del self.encoder.encoder.fc

        # 1x1 convolution for channel reduction in multi-layer feature map fusion
        enc_feat_dim = sum(self.encoder.num_ch_enc[fusion_level:])
        self.conv1x1 = conv2d(enc_feat_dim, fusion_feat_in_dim, kernel_size=1, padding_mode='reflect')

        # fusion net
        fusion_feat_out_dim = self.encoder.num_ch_enc[fusion_level]
        self.fusion_net = DepthVFNet(
            fusion_feat_in_dim,
            fusion_feat_out_dim,
            input_width=input_width,
            input_height=input_height,
        )

        # depth decoder
        num_ch_enc = self.encoder.num_ch_enc[:(fusion_level + 1)]
        num_ch_dec = [16, 32, 64, 128, 256]
        self.decoder = DepthDecoder(fusion_level, num_ch_enc, num_ch_dec, use_skips=use_skips)
        self.fusion_level = fusion_level

    def forward(
        self,
        images,
        mask,
        intrinsic,
        extrinsic,
        inv_intrinsic=None,
        inv_extrinsic=None,
        distortion=None,
        fov=None,
    ):
        # images (batch_size x num_cams x channels x height x width)
        batch_size, num_cams, _, _, _ = images.shape

        # ((batch_size x num_cams) x channels x height x width)
        packed_input = pack_cam_feat(images)

        # feature encoder
        packed_feats = self.encoder(packed_input)
        # aggregate feature H / 2^(lev+1) x W / 2^(lev+1)
        _, _, up_h, up_w = packed_feats[self.fusion_level].size()

        packed_feats_list = (
            packed_feats[self.fusion_level:self.fusion_level + 1] + [
                F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True)
                for feat in packed_feats[self.fusion_level + 1:]
            ]
        )

        # packed_feats_agg ((batch_size x num_cams) x fusion_feat_in_dim x feat_width x feat_height)
        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1))
        # feats_agg (batch_size x num_cams x fusion_feat_in_dim x feat_width x feat_height)
        feats_agg = unpack_cam_feat(packed_feats_agg, batch_size, num_cams)

        # fusion_net, backproject each feature into the 3D voxel space
        voxel_feat = self.fusion_net(
            mask=mask,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            feats_agg=feats_agg,
            inv_intrinsic=inv_intrinsic,
            inv_extrinsic=inv_extrinsic,
            distortion=distortion,
            fov=fov,
        )

        feat_in = packed_feats[:self.fusion_level] + [voxel_feat]
        packed_depth_outputs = self.decoder(feat_in)

        # depth_outputs (batch_size x num_cams x 1 x height x width)
        depth_outputs = unpack_cam_feat(packed_depth_outputs, batch_size, num_cams)

        return depth_outputs
