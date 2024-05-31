from typing import Callable

import torch


def lazy_bmm(a: torch.Tensor, b: torch.Tensor, lazy_batch_size: int) -> torch.Tensor:
    index_list = list(range(a.shape[0]))
    chunks = [index_list[i:i + lazy_batch_size] for i in range(0, len(index_list), lazy_batch_size)]
    outputs = []
    for chunk in chunks:
        outputs.append(torch.bmm(a[chunk], b[chunk]))
    return torch.cat(outputs, dim=0)


def batch_first_single_output_wrapper(a: torch.Tensor,
                                      fn: Callable,
                                      lazy_batch_size: int,
                                      **kwargs) -> torch.Tensor:
    index_list = list(range(a.shape[0]))
    chunks = [index_list[i:i + lazy_batch_size] for i in range(0, len(index_list), lazy_batch_size)]
    outputs = []
    for chunk in chunks:
        outputs.append(fn(a[chunk], **kwargs))
    return torch.cat(outputs, dim=0)
