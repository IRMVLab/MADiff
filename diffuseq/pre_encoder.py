# Defines feature encoders for GLIP, motion, location, and fused inputs.
import torch.nn as nn

from .utils.nn import linear


def _flatten_bt(x):

    batch_size, time_steps, feat_dim = x.shape
    return batch_size, time_steps, x.view(batch_size * time_steps, feat_dim)


def _restore_bt(x, batch_size, time_steps):

    return x.view(batch_size, time_steps, x.shape[-1])


def _encode_bt(x, encoder):

    batch_size, time_steps, x = _flatten_bt(x)
    x = encoder(x)
    return _restore_bt(x, batch_size, time_steps)


class PreEncoder(nn.Module):


    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim = encoder_hidden_dims
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            linear(input_dims, output_dims),
        )

    def forward(self, x):
        return _encode_bt(x, self.feat_embed)


class MotionEncoder(nn.Module):


    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim = encoder_hidden_dims
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            linear(input_dims, output_dims, bias=False),
        )

    def forward(self, x):
        return _encode_bt(x, self.feat_embed)


class LocEncoder(nn.Module):


    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., out_layer=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.out_layer = out_layer() if out_layer is not None else None

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.out_layer is not None:
            x = self.out_layer(x)
        return x


class GLIPEncoder(nn.Module):


    def __init__(
        self,
        input_dims_conv,
        output_dims_conv,
        input_dims,
        output_dims,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.input_dims_conv = input_dims_conv
        self.output_dims_conv = output_dims_conv

        self.conv1x1 = nn.Conv2d(
            input_dims_conv, output_dims_conv, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        self.feat_embed = nn.Sequential(
            linear(input_dims, output_dims, bias=False),
        )

    def forward(self, x):

        batch_size, time_steps, channels, height, width = x.shape
        x = x.reshape(batch_size * time_steps, channels, height, width)
        x = self.conv1x1(x)


        x = x.reshape(batch_size * time_steps, self.output_dims_conv, -1)
        x = x.reshape(batch_size * time_steps * self.output_dims_conv, -1)
        x = self.feat_embed(x)
        x = x.reshape(batch_size, time_steps, -1)
        return x
