# Provides neural-network helper layers and timestep embeddings.


import math

import torch as th
import torch.nn as nn


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class SIG(nn.Module):
    def forward(self, x):
        return th.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def linear(*args, **kwargs):

    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):

    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):

    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):

    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):

    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):

    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):

    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):

    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)

    args = timesteps[:, None].float() * freqs[None]

    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
