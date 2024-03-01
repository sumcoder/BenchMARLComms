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
    return loss.mean()  # loss.sum(1).squeeze()  # out.cpu()


class SoftmaxAttention(BaseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = self.qk_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        print("Vanilla tx!")

    def dist_attend(self, q, k, v):
        info = {}
        a, b, h, n, _ = k.shape

        dots = einsum("a b h i d, a b h j d -> a b h i j", q, k) * self.scale
        weights = self.softmax(dots)
        out = einsum("a b h i j, a b h j d -> a b h i d", weights, v)

        # info["k_reg_loss"] = component_log_loss(k, 2 * NOISE_WIDTH)
        # info["wv_reg_loss"] = 6 #component_log_loss(k, 2 * NOISE_WIDTH)

        # out = v

        return out, info


class NoisySoftmaxAttention(BaseAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = self.qk_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        print("==Entering Noisy channels!==")

    def dist_attend(self, q, k, v):
        a, b, h, n, _ = k.shape

        info = {}
        info["k_reg_loss"] = component_log_loss(k, 2 * NOISE_WIDTH)
        k = k + torch_uniform_like(k, NOISE_WIDTH)

        dots = einsum("a b h i d, a b h j d -> a b h i j", q, k) * self.scale
        weights = self.softmax(dots)

        out = einsum("a b h i j, a b h j d -> a b h i d", weights, v)

        # wv = einsum("a b h i j, a b h j d -> a b h i j d", weights, v)

        info["wv_reg_loss"] = 12 #component_log_loss(wv, 2 * NOISE_WIDTH)

        # repeat_shape = len(k_mat.shape)
        # if repeat_shape == 2:
        #     rep = [1, n, 1]
        # else:
        #     rep = [n, 1]
        # info['bits_mat'] = (k_mat.unsqueeze(-2).repeat(*rep) + wv_mat)#.cpu().detach().numpy()

        # wv = wv + torch_uniform_like(wv, NOISE_WIDTH)
        # out = wv.sum(-2)

        return out, info
