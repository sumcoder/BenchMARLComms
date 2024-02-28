import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
from collections import defaultdict

from .softmax_attn import NoisySoftmaxAttention, SoftmaxAttention

from .registry import attention_registry

# helpers

def dict_merge(list_of_dict, dim=0, mode="concat", backend="torch"):
    ret = defaultdict(list)
    for d in list_of_dict:
        for k, v in d.items():
            ret[k].append(v)
    if mode == "concat":
        if backend == "torch":
            ret = {k: torch.cat(v, dim) for k, v in ret.items()}
        else:
            ret = {k: np.concatenate(v, axis=dim) for k, v in ret.items()}
    elif mode == "stack":
        if backend == "torch":
            ret = {k: torch.stack(v, dim) for k, v in ret.items()}
        else:
            ret = {k: np.stack(v, axis=dim) for k, v in ret.items()}
    elif mode == "mean":
        if backend == "torch":
            ret = {k: torch.mean(torch.stack(v, 0), 0) for k, v in ret.items()}
        else:
            ret = {k: np.mean(v) for k, v in ret.items()}
    return ret


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim).cuda()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        ).cuda()

    def forward(self, x):
        return self.net(x)


def get_attention_layer(name):
    if name in attention_registry:
        return attention_registry[name]
    else:
        raise NotImplementedError


class Transformer(nn.Module):
    def __init__(self, comms_penalty, dim, depth, heads, qk_dim, v_dim, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.bits_mat = 0
        # AttentionLayer = get_attention_layer(attention)
        AttentionLayer = SoftmaxAttention if not comms_penalty else NoisySoftmaxAttention
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, AttentionLayer(dim=dim, qk_dim=qk_dim, v_dim=v_dim, heads=heads,
                                                           dropout=dropout),),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, dist=False):
        infos = []
        for attn, ff in self.layers:
            x_, info = attn(x, dist=dist)
            x = x_ + x
            x = ff(x) + x
            infos.append(info)

            # if 'bits_mat' in info:
            #     self.bits_mat = info['bits_mat'].cpu().detach().numpy()

        # info = dict_merge(infos, mode="mean")
        return x, info