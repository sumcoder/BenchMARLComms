import math
import torch
import numpy as np
import torch.nn.functional as F


def torch_uniform_like(x, width):
    """
    Generate tensor of same shape as `x` from random uniform distribution
    between -`width` and `width`

    This operation is differentiable with respect to `width`
    """
    # unit_sample = torch.rand_like(x)
    unit_sample = np.random.randn(*x.shape)
    unit_sample = torch.Tensor(unit_sample).to('cuda')
    sample = (unit_sample - 0.5) * 2 * width
    return sample


def torch_layer_norm(x, norm=1, ord=2, eps=1e-5):
    """
    Scale tensor so that each vector in the last dimension will have norm `norm`
    """
    return x / (torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True) + eps) * norm


def topk_softmax(x, k):
    topk_indices = torch.topk(x, k, dim=-1).indices
    m = torch.full_like(x, -float("inf"))
    m.scatter_(-1, topk_indices, 0)
    return torch.softmax(x+m, -1)


if __name__ == '__main__':
    # print(topk_softmax(torch.randn(16, 16, 6), 2)[0, 0])
    weights = torch.ones(2, 4)
    weights[0, :] = 0
    print(torch.all(weights == 0, -1).float().mean())