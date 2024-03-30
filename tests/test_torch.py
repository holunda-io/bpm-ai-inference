import torch


def test_torch():
    x = torch.rand(3, 3)
    assert x.dim() == 2
