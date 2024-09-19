from typing import Any, Sequence, Tuple, Union

import torch
from ignite.utils import convert_tensor


def prepare_batch(
    batch: Sequence[Any],
    device: Union[str, torch.device] = None,
    non_blocking: bool = False,
) -> Tuple[Any, Any]:
    batch = list(batch)

    for i, item in enumerate(batch):
        batch[i] = convert_tensor(item, device=device, non_blocking=non_blocking)

    return (batch, (batch,))
