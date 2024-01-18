import numpy as np
import torch
from einops import rearrange
from torch import einsum, nn

from .base_attn import BaseAttention
from .registry import register_attention
from .torch_utils import torch_uniform_like


# constants
NOISE_WIDTH = 1.0 / 15
DEBUG = True

def inject_uniform_noise(x, width):
    '''
    x: shape (batch, head, number of nodes, hidden dim)
    '''
    noise = torch_uniform_like(x, width)
    x = x + noise.sum(2, keepdim=True)
    return x


def component_log_loss(x, delta, reduction="mean"):
    """
    Log loss to penalize the number of bits communicated during training
    """
    # print("input shape for loss", x.shape)
    loss = torch.log2(2 * x.abs() / delta + 1).sum(-1)
    # print("loss shape", loss.shape)
    # msg_shape = len(x.shape)
    # if msg_shape == 4:  # message type: key
    #     out = loss[0, 0]  # 1D vector
    # else:  # message type: weighted value
    #     out = loss[0, 0]  # 2D matrix
    if reduction == "none":
        return loss
    return loss.mean(), loss.sum(1).squeeze()  # out.cpu()


class SoftmaxAttention(BaseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = self.qk_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def dist_attend(self, q, k, v):
        info = {}
        a, b, h, n, _ = k.shape

        dots = einsum("a b h i d, a b h j d -> a b h i j", q, k) * self.scale
        weights = self.softmax(dots)
        out = einsum("a b h i j, a b h j d -> a b h i d", weights, v)

        # out = v

        return out, info


class NoisySoftmaxAttention(BaseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = self.qk_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def dist_attend(self, q, k, v):
        b, h, n, _ = q.shape
        info = {}
        info["k_reg_loss"], k_mat = component_log_loss(k, 2 * NOISE_WIDTH)
        # print(k_mat.shape, "important key")
        # print(k[0,0,0,:5])
        k = k + torch_uniform_like(k, NOISE_WIDTH)

        dots = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        weights = self.softmax(dots)

        wv = einsum("b h i j, b h j d -> b h i j d", weights, v)
        # print(wv[0,0,0,0,:5])
        info["wv_reg_loss"], wv_mat = component_log_loss(wv, 2 * NOISE_WIDTH)
        # print(info['wv_mat'].shape, "important wv")

        repeat_shape = len(k_mat.shape)
        if repeat_shape == 2:
            rep = [1, n, 1]
        else:
            rep = [n, 1]
        # info['bits_mat'] = (k_mat.unsqueeze(-2).repeat(*rep) + wv_mat)#.cpu().detach().numpy()

        wv = wv + torch_uniform_like(wv, NOISE_WIDTH)
        out = wv.sum(-2)

        return out, info