import torch

import utils
from models import VFDepthAlgo
from models.vf_depth.vf_depth import VFDepth
from models.geometry.geometry_util import vec_to_matrix


if __name__ == '__main__':
    cfg = utils.get_config('./configs/nuscenes/nusc_surround_fusion.yaml', mode='train')

    org_model = VFDepthAlgo(cfg, 0)
    org_model.load_weights()
    org_model.set_val()
    model = VFDepth()
    weight = torch.load('./output/weights/vfdepth.pt')
    model.load_state_dict(weight)
    model.cuda()
    model.eval()

    dataset = org_model.train_dataloader()
    sample = next(iter(dataset))

    model.pose_net.encoder.load_state_dict(org_model.models['pose_net'].encoder.state_dict())
    model.pose_net.conv1x1.load_state_dict(org_model.models['pose_net'].conv1x1.state_dict())
    model.pose_net.fusion_net.load_state_dict(org_model.models['pose_net'].fusion_net.state_dict())
    model.pose_net.decoder.load_state_dict(org_model.models['pose_net'].pose_decoder.state_dict())

    model.depth_net.encoder.load_state_dict(org_model.models['depth_net'].encoder.state_dict())
    model.depth_net.conv1x1.load_state_dict(org_model.models['depth_net'].conv1x1.state_dict())
    model.depth_net.fusion_net.load_state_dict(org_model.models['depth_net'].fusion_net.state_dict())
    model.depth_net.decoder.load_state_dict(org_model.models['depth_net'].decoder.state_dict())

    with torch.no_grad():
        # Pose
        f_i = 1
        outputs, _ = org_model.process_batch(sample, 0)
        org_cam_T_cams = [outputs['cam', cam]['cam_T_cam', 0, f_i] for cam in range(6)]

        frame_ids = [0, 1]
        cur_image = sample[('color_aug', frame_ids[0], 0)]
        next_image = sample[('color_aug', frame_ids[1], 0)]

        fusion_level = 2
        mask = sample['mask']
        intrinsic = sample['K', fusion_level + 1]
        inv_intrinsic = sample['inv_K', fusion_level + 1]
        extrinsic = sample['extrinsics_inv']
        inv_extrinsic = sample['extrinsics']

        axis_angle, translation = model.pose_net(cur_image, next_image, mask, intrinsic, extrinsic)
        cam_T_cam = vec_to_matrix(axis_angle[:, 0], translation[:, 0], invert=(f_i < 0))
        exts = inv_extrinsic
        exts_inv = extrinsic
        ref_ext = exts[:, 0, ...]
        ref_ext_inv = exts_inv[:, 0, ...]

        ref_T = cam_T_cam
        cam_T_cams = []
        for cam in range(6):
            cur_ext = exts[:, cam, ...]
            cur_ext_inv = exts_inv[:, cam, ...]
            cur_T = cur_ext_inv @ ref_ext @ ref_T @ ref_ext_inv @ cur_ext
            cam_T_cams.append(cur_T)

        # compare org_cam_T_cams and cam_T_cams
        for org_T, T in zip(org_cam_T_cams, cam_T_cams):
            print(torch.all(torch.abs(org_T - T) < 1e-9))

        # Depth
        images = sample[('color_aug', 0, 0)]
        depth_maps = model.depth_net(images, mask, intrinsic, inv_intrinsic, extrinsic, inv_extrinsic)

        for i in range(6):
            print(torch.all(abs(depth_maps[:, i] - outputs[('cam', i)]['disp', 0]) < 1e-9))
