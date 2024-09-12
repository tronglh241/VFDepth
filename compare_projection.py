import torch

from flame_based.models.vf_depth import VFDepth

if __name__ == '__main__':
    model = VFDepth()
    model.eval()
    feats_agg = torch.rand(2, 6, 16, 300, 400)
    input_mask = torch.rand(2, 6, 1, 300, 400).round()
    intrinsic = torch.rand(2, 6, 4, 4)
    extrinsic = torch.rand(2, 6, 4, 4)
    intrinsic[..., -2:] = 0
    intrinsic[..., -2:, :] = 0
    intrinsic[..., -1, -1] = 1
    intrinsic[..., -2, -2] = 1
    extrinsic[..., -1:] = 0
    extrinsic[..., -1:, :] = 0
    extrinsic[..., -1, -1] = 1
    output1 = model.pose_net.fusion_net.backproject_into_voxel(feats_agg, input_mask, intrinsic, extrinsic)
    output2 = model.pose_net.fusion_net.backproject_into_voxel2(feats_agg, input_mask, intrinsic, extrinsic)
    # for i in range(6):
    #     print(output1[0][i].mean(), output2[0][i].mean())
    #     print(output1[0][i].min(), output2[0][i].min())
    #     print(output1[0][i].max(), output2[0][i].max())
    #     print(output1[0][i].std(), output2[0][i].std())

    #     print(torch.mean(abs(output1[0][i] - output2[0][i])))
    #     print(torch.mean(abs(output1[1][i].float() - output2[1][i].float())))
    # breakpoint()
    voxel_feat = torch.rand(2, 64, 100 * 100 * 20)
    inv_intrinsic = torch.inverse(intrinsic)
    inv_extrinsic = torch.inverse(extrinsic)
    output1 = model.depth_net.fusion_net.project_voxel_into_image(voxel_feat, intrinsic, extrinsic)
    output2 = model.depth_net.fusion_net.project_voxel_into_image2(voxel_feat, intrinsic, extrinsic)

    for i in range(6):
        print(torch.mean(abs(output1[i] - output2[i])))
