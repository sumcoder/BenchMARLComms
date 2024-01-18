import torch
from einops import rearrange
from torch import nn


class BaseAttention(nn.Module):
    def __init__(self, *, dim=512, qk_dim=64, v_dim=64, heads=8, dropout=0.0, attention_config=None, critic_truncate_len=0):
        super().__init__()
        inner_qk_dim = qk_dim * heads
        inner_v_dim = v_dim * heads

        self.heads = heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.critic_truncate_len = critic_truncate_len

        self.key_mat = 0

        if not critic_truncate_len:
            self.to_qk = nn.Linear(dim, inner_qk_dim * 2, bias=False).cuda()
            self.to_v = nn.Linear(dim, inner_v_dim, bias=False).cuda()
        else:
            self.to_q = nn.Linear(dim, inner_qk_dim, bias=False)
            self.to_k = nn.Linear(dim - critic_truncate_len, inner_qk_dim, bias=False)
            self.to_v = nn.Linear(dim - critic_truncate_len, inner_v_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_v_dim, dim - critic_truncate_len),
                                    nn.Dropout(dropout)).cuda()

    def attend(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, dist=False):
        # print("input to base attention", x.shape)
        l = len(x.shape)
        if l == 3:
            x = x.unsqueeze(0)
        b1, b2, n, _, h = *x.shape, self.heads

        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = map(lambda t: rearrange(t, "b1 b2 n (h d) -> b1 b2 h n d", h=h), qk)
        v = rearrange(self.to_v(x), "b1 b2 n (h d) -> b1 b2 h n d", h=h)

        # print("k q v shapes", k.shape, q.shape, v.shape)
        # if dist:
        out, info = self.dist_attend(q, k, v)

        self.key_mat = 0
        # else:
        #     out, info = self.attend(q, k, v)

        # print("after dist atted", out.shape)

        out = rearrange(out, "b1 b2 h n d -> b1 b2 n (h d)")
        if l == 3:
            out = out[0]
        out = self.to_out(out)

        return out, info

    def dist_attend(self, *args, **kwargs):
        raise NotImplementedError