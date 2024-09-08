import torch
import torch.nn.functional as F
from torch import nn

from external.layers import PoseDecoder, ResnetEncoder
from models.geometry.geometry_util import vec_to_matrix
from models.vf_depth.vf_net import PoseVFNet
from network.blocks import conv2d, pack_cam_feat, unpack_cam_feat


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
        self.fusion_net = PoseVFNet(fusion_feat_in_dim, fusion_feat_out_dim)

        # depth decoder
        self.decoder = PoseDecoder(
            num_ch_enc=[fusion_feat_out_dim],
            num_input_features=1,
            num_frames_to_predict_for=1,
            stride=2,
        )
        self.fusion_level = fusion_level

    def forward(self, cur_image, next_image, mask, intrinsic, extrinsic):
        # cur_image (batch_size x num_cams x channels x height x width)
        # next_image (batch_size x num_cams x channels x height x width)
        assert cur_image.shape == next_image.shape
        batch_size, num_cams, _, _, _ = cur_image.shape

        # images (batch_size x num_cams x (channels x 2) x height x width)
        images = torch.cat([cur_image, next_image], 2)
        # packed_pose_images ((batch_size x num_cams) x (channels x 2) x height x width)
        packed_pose_images = pack_cam_feat(images)

        packed_feats = self.encoder(packed_pose_images)

        # aggregate feature H / 2^(lev + 1) x W / 2^(lev + 1)
        _, _, up_h, up_w = packed_feats[self.fusion_level].size()

        packed_feats_list = (
            packed_feats[self.fusion_level:self.fusion_level + 1] + [
                F.interpolate(feat, [up_h, up_w], mode='bilinear', align_corners=True)
                for feat in packed_feats[self.fusion_level + 1:]
            ]
        )

        # packed_feats_agg ((batch_size x num_cams) x fusion_feat_in_dim x feat_height x feat_width)
        packed_feats_agg = self.conv1x1(torch.cat(packed_feats_list, dim=1))
        # feats_agg (batch_size x num_cams x fusion_feat_in_dim x feat_height x feat_width)
        feats_agg = unpack_cam_feat(packed_feats_agg, batch_size, num_cams)

        # fusion_net, backproject each feature into the 3D voxel space
        bev_feat = self.fusion_net(
            mask,
            intrinsic,
            extrinsic,
            feats_agg,
        )
        axis_angle, translation = self.decoder([[bev_feat]])
        return axis_angle, torch.clamp(translation, -4.0, 4.0)  # for DDAD dataset

    def compute_pose(
        self,
        axis_angle,
        translation,
        invert=False,
    ):
        return vec_to_matrix(axis_angle[:, 0], translation[:, 0], invert=invert)

    def compute_poses(
        self,
        axis_angle,
        translation,
        invert,
        ref_extrinsic,
        ref_inv_extrinsic,
        extrinsic,
        inv_extrinsic,
    ):
        ref_T = self.compute_pose(axis_angle, translation, invert)
        poses = extrinsic @ ref_inv_extrinsic @ ref_T.unsqueeze(1) @ ref_extrinsic @ inv_extrinsic
        return poses
