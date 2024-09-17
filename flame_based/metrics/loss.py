from collections import defaultdict
from typing import Callable, Dict, Sequence, Tuple, Union, cast

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric


class Loss(Metric):
    def __init__(
        self,
        loss_fn: Callable,
        output_transform: Callable = lambda x: x,
        batch_size: Callable = len,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Loss, self).__init__(output_transform, device=device, skip_unrolling=True)
        self._loss_fn = loss_fn
        self._batch_size = batch_size

    def reset(self) -> None:
        self._sum_losses = defaultdict(float)
        self._num_examples = 0

    def update(self, output: Sequence[Union[torch.Tensor, Dict]]) -> None:
        if len(output) == 2:
            y_pred, y = cast(Tuple[torch.Tensor, torch.Tensor], output)
            kwargs: Dict = {}
        else:
            y_pred, y, kwargs = cast(Tuple[torch.Tensor, torch.Tensor, Dict], output)
        average_loss = self._loss_fn(y_pred, y, **kwargs).detach()

        if len(average_loss.shape) != 0:
            raise ValueError("loss_fn did not return the average loss.")

        n = self._batch_size(y)
        self._sum_losses['total_loss'] += average_loss.to(self._device) * n

        for k, v in self._loss_fn.loss_mean:
            self._sum_losses[k] += v.to(self._device) * n

        self._num_examples += n

    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Loss must have at least one example before it can be computed.")

        for k, v in self._sum_losses.items():
            self._sum_losses[k] = v / self._num_examples
            self._sum_losses[k] = self._sum_losses.item()

        return self._sum_losses
