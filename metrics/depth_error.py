from collections import defaultdict
from typing import Callable

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class DepthError(Metric):
    _state_dict_all_req_keys = ("_sum", "_num_examples")

    def __init__(
        self,
        min_eval_depth: float,
        max_eval_depth: float,
        compute_true_depth_maps: Callable,
        output_transform: Callable = lambda x: x,
    ):
        self.min_eval_depth = min_eval_depth
        self.max_eval_depth = max_eval_depth
        self.metric_names = ['abs_rel', 'sq_rel', 'rms', 'log_rms', 'a1', 'a2', 'a3']
        self.compute_true_depth_maps = compute_true_depth_maps
        super(DepthError, self).__init__(output_transform, skip_unrolling=True)

    @reinit__is_reduced
    def reset(self) -> None:
        self.abs_rel = 0
        self.sq_rel = 0
        self.rmse = 0
        self.rmse_log = 0
        self.a1 = 0
        self.a2 = 0
        self.a3 = 0
        self.abs_rel_median = 0
        self.sq_rel_median = 0
        self.rmse_median = 0
        self.rmse_log_median = 0
        self.a1_median = 0
        self.a2_median = 0
        self.a3_median = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output) -> None:
        depth_maps, gt_true_depth_maps, masks, intrinsics = output
        true_depth_maps = self.compute_true_depth_maps(depth_maps, intrinsics)
        med_scale = []

        error_metric_dict = defaultdict(float)
        error_median_dict = defaultdict(float)

        for cam in range(depth_maps.shape[1]):
            depth_gt = gt_true_depth_maps[:, cam]
            depth_pred = true_depth_maps[:, cam]
            depth_pred = torch.clamp(depth_pred, self.min_eval_depth, self.max_eval_depth)
            mask = (depth_gt > self.min_eval_depth) * (depth_gt < self.max_eval_depth) * masks[:, cam]
            mask = mask.bool()

            depth_gt = depth_gt[mask]
            depth_pred = depth_pred[mask]

            # calculate median scale
            scale_val = torch.median(depth_gt) / torch.median(depth_pred)
            med_scale.append(round(scale_val.cpu().numpy().item(), 2))

            depth_pred_metric = torch.clamp(depth_pred, min=self.min_eval_depth, max=self.max_eval_depth)
            depth_errors_metric = self.cal_depth_error(depth_pred_metric, depth_gt)

            depth_pred_median = torch.clamp(depth_pred * scale_val, min=self.min_eval_depth, max=self.max_eval_depth)
            depth_errors_median = self.cal_depth_error(depth_pred_median, depth_gt)

            for i in range(len(depth_errors_metric)):
                key = self.metric_names[i]
                error_metric_dict[key] += depth_errors_metric[i]
                error_median_dict[key] += depth_errors_median[i]

        for key in error_metric_dict.keys():
            error_metric_dict[key] = error_metric_dict[key].cpu().numpy() / depth_maps.shape[1]
            error_median_dict[key] = error_median_dict[key].cpu().numpy() / depth_maps.shape[1]

        n = depth_maps.shape[0]
        self.abs_rel += error_metric_dict['abs_rel'] * n
        self.sq_rel += error_metric_dict['sq_rel'] * n
        self.rmse += error_metric_dict['rmse'] * n
        self.rmse_log += error_metric_dict['rmse_log'] * n
        self.a1 += error_metric_dict['a1'] * n
        self.a2 += error_metric_dict['a2'] * n
        self.a3 += error_metric_dict['a3'] * n
        self.abs_rel_median += error_median_dict['abs_rel'] * n
        self.sq_rel_median += error_median_dict['sq_rel'] * n
        self.rmse_median += error_median_dict['rmse'] * n
        self.rmse_log_median += error_median_dict['rmse_log'] * n
        self.a1_median += error_median_dict['a1'] * n
        self.a2_median += error_median_dict['a2'] * n
        self.a3_median += error_median_dict['a3'] * n

        self._num_examples += n

    @sync_all_reduce(
        'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3',
        'abs_rel_median', 'sq_rel_median', 'rmse_median', 'rmse_log_median', 'a1_median', 'a2_median', 'a3_median',
        '_num_examples',
    )
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Loss must have at least one example before it can be computed.")

        self.abs_rel = float(self.abs_rel / self._num_examples)
        self.sq_rel = float(self.sq_rel / self._num_examples)
        self.rmse = float(self.rmse / self._num_examples)
        self.rmse_log = float(self.rmse_log / self._num_examples)
        self.a1 = float(self.a1 / self._num_examples)
        self.a2 = float(self.a2 / self._num_examples)
        self.a3 = float(self.a3 / self._num_examples)
        self.abs_rel_median = float(self.abs_rel_median / self._num_examples)
        self.sq_rel_median = float(self.sq_rel_median / self._num_examples)
        self.rmse_median = float(self.rmse_median / self._num_examples)
        self.rmse_log_median = float(self.rmse_log_median / self._num_examples)
        self.a1_median = float(self.a1_median / self._num_examples)
        self.a2_median = float(self.a2_median / self._num_examples)
        self.a3_median = float(self.a3_median / self._num_examples)

        return {
            'abs_rel': self.abs_rel,
            'sq_rel': self.sq_rel,
            'rmse': self.rmse,
            'rmse_log': self.rmse_log,
            'a1': self.a1,
            'a2': self.a2,
            'a3': self.a3,
            'abs_rel_median': self.abs_rel_median,
            'sq_rel_median': self.sq_rel_median,
            'rmse_median': self.rmse_median,
            'rmse_log_median': self.rmse_log_median,
            'a1_median': self.a1_median,
            'a2_median': self.a2_median,
            'a3_median': self.a3_median,
        }

    def cal_depth_error(self, pred, target):
        """
        This function calculates depth error using various metrics.
        """
        abs_rel = torch.mean(torch.abs(pred - target) / target)
        sq_rel = torch.mean((pred - target).pow(2) / target)
        rmse = torch.sqrt(torch.mean((pred - target).pow(2)))
        rmse_log = torch.sqrt(torch.mean((torch.log(target) - torch.log(pred)).pow(2)))

        thresh = torch.max((target / pred), (pred / target))
        a1 = (thresh < 1.25).float().mean()
        a2 = (thresh < 1.25**2).float().mean()
        a3 = (thresh < 1.25**3).float().mean()
        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
