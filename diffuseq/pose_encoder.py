# Defines post-encoder modules that decode trajectory features.
import torch.nn as nn

from .utils.nn import linear


def _flatten_bt(x):

    batch_size, time_steps, feat_dim = x.shape
    return batch_size, time_steps, x.view(batch_size * time_steps, feat_dim)


def _restore_bt(x, batch_size, time_steps):

    return x.view(batch_size, time_steps, x.shape[-1])


class PostEncoder(nn.Module):


    def __init__(
        self,
        input_dims,
        output_dims,
        encoder_hidden_dims1,
        encoder_hidden_dims2,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.hidden_t_dim1 = encoder_hidden_dims1
        self.hidden_t_dim2 = encoder_hidden_dims2
        self.output_dims = output_dims

        self.feat_embed = nn.Sequential(
            linear(input_dims, encoder_hidden_dims1, bias=False),
            linear(encoder_hidden_dims1, encoder_hidden_dims2, bias=False),
            linear(encoder_hidden_dims2, output_dims, bias=False),
        )

    def forward(self, x):
        batch_size, time_steps, x = _flatten_bt(x)
        x = self.feat_embed(x)
        return _restore_bt(x, batch_size, time_steps)
