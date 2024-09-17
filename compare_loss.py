import torch

import utils
from flame_based.losses.multi_cam_loss import MultiCamLoss
from models import VFDepthAlgo
_NO_DEVICE_KEYS = ['idx', 'dataset_idx', 'sensor_name', 'filename']

if __name__ == '__main__':
    cfg = utils.get_config('./configs/nuscenes/nusc_surround_fusion.yaml', mode='train')

    org_model = VFDepthAlgo(cfg, 0)
    org_model.load_weights()
    org_model.set_val()

    dataset = org_model.train_dataloader()
    inputs = next(iter(dataset))

    multi_cam_loss = MultiCamLoss(
        disparity_smoothness=0.001,
        spatio_coeff=0.03,
        spatio_tempo_coeff=0.1,
    )

    with torch.no_grad():
        for key, ipt in inputs.items():
            if key not in _NO_DEVICE_KEYS:
                if 'context' in key:
                    inputs[key] = [ipt[k].float().to(0) for k in range(len(inputs[key]))]
                else:
                    inputs[key] = ipt.float().to(0)

        outputs = org_model.estimate_vfdepth(inputs)
        breakpoint()
        for cam in range(org_model.num_cams):
            org_model.pred_cam_imgs(inputs, outputs, cam)
            org_cam_loss, org_loss_dict = org_model.losses(inputs, outputs, cam)
            outputs['cam', cam]['color', -1] = outputs['cam', cam]['color', -1, 0]
            outputs['cam', cam]['color', 1] = outputs['cam', cam]['color', 1, 0]
            outputs['cam', cam]['overlap_mask', 0] = outputs['cam', cam]['overlap_mask', 0, 0]
            outputs['cam', cam]['overlap_mask', -1] = outputs['cam', cam]['overlap_mask', -1, 0]
            outputs['cam', cam]['overlap_mask', 1] = outputs['cam', cam]['overlap_mask', 1, 0]
            outputs['cam', cam]['overlap', 0] = outputs['cam', cam]['overlap', 0, 0]
            outputs['cam', cam]['overlap', -1] = outputs['cam', cam]['overlap', -1, 0]
            outputs['cam', cam]['overlap', 1] = outputs['cam', cam]['overlap', 1, 0]
            cam_loss, loss_dict = multi_cam_loss.forward(
                cam_org_prev_image=inputs['color', -1, 0][:, cam],
                cam_org_image=inputs['color', 0, 0][:, cam],
                cam_org_next_image=inputs['color', 1, 0][:, cam],
                cam_target_view=outputs['cam', cam],
                cam_depth_map=outputs['cam', cam]['disp', 0],
                cam_mask=inputs['mask'][:, cam],
            )
            breakpoint()
