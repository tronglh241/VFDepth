from collections import defaultdict
from typing import Callable, Dict, Sequence, Tuple, Union, cast

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class Loss(Metric):
    _state_dict_all_req_keys = ("_sum", "_num_examples")

    def __init__(
        self,
        loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Loss, self).__init__(output_transform, device=device, skip_unrolling=True)
        self._loss_fn = loss_fn

    @reinit__is_reduced
    def reset(self) -> None:
        self.total_loss = torch.tensor(0.0, device=self._device)
        self.reproj_loss = torch.tensor(0.0, device=self._device)
        self.spatio_loss = torch.tensor(0.0, device=self._device)
        self.spatio_tempo_loss = torch.tensor(0.0, device=self._device)
        self.smooth_loss = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, Dict]]) -> None:
        if len(output) == 2:
            y_pred, y = cast(Tuple[torch.Tensor, torch.Tensor], output)
            kwargs: Dict = {}
        else:
            y_pred, y, kwargs = cast(Tuple[torch.Tensor, torch.Tensor, Dict], output)
        average_loss = self._loss_fn(y_pred, y, **kwargs).detach()

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = y[0].shape[0]
        self.total_loss += average_loss.to(self._device) * n
        self.reproj_loss += self._loss_fn.loss_fn.loss_mean['reproj_loss']
        self.spatio_loss += self._loss_fn.loss_fn.loss_mean['spatio_loss']
        self.spatio_tempo_loss += self._loss_fn.loss_fn.loss_mean['spatio_tempo_loss']
        self.smooth_loss += self._loss_fn.loss_fn.loss_mean['smooth']

        self._num_examples += n

    @sync_all_reduce("total_loss", "reproj_loss", "spatio_loss", "spatio_tempo_loss", "smooth_loss", "_num_examples")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Loss must have at least one example before it can be computed.")

        self.total_loss = float(self.total_loss / self._num_examples)
        self.reproj_loss = float(self.reproj_loss / self._num_examples)
        self.spatio_loss = float(self.spatio_loss / self._num_examples)
        self.spatio_tempo_loss = float(self.spatio_tempo_loss / self._num_examples)
        self.smooth_loss = float(self.smooth_loss / self._num_examples)

        return {
            'total_loss': self.total_loss,
            'reproj_loss': self.reproj_loss,
            'spatio_loss': self.spatio_loss,
            'spatio_tempo_loss': self.spatio_tempo_loss,
            'smooth_loss': self.smooth_loss,
        }
