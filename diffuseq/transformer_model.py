# Defines the Mamba denoising model used by diffusion.
import torch
import torch.nn as nn

from mambapy.mamba import Mamba, MambaConfig

from .utils.nn import SiLU, linear, timestep_embedding


class HOIMamba(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_layers=1,
        output_dims=512,
        hidden_t_dim=512,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        config = MambaConfig(d_model=self.d_model, n_layers=self.n_layers)
        self.mamba_encoder = Mamba(config)

        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.hidden_size = 512

        time_embed_dim = hidden_t_dim * 2
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, self.hidden_size),
        )

        self.register_buffer("position_ids", torch.arange(1000).expand((1, -1)))
        self.position_embeddings = nn.Embedding(1000, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size)

        if self.output_dims != self.hidden_size:
            self.output_down_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_dims),
            )

    def forward(self, x_r, timesteps, motion_feat_encoded, valid_mask=None):
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        emb_xr = x_r

        seq_length = x_r.size(1)
        position_ids = self.position_ids[:, :seq_length]
        emb_inputs_r = (
            self.position_embeddings(position_ids)
            + emb_xr
            + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        )
        emb_inputs_r = self.LayerNorm(emb_inputs_r)

        motion_length = motion_feat_encoded.size(1)
        position_ids_motion = self.position_ids[:, :motion_length]
        emb_motion = self.position_embeddings(position_ids_motion) + motion_feat_encoded
        emb_motion = self.LayerNorm(emb_motion)

        emb_inputs_r = self.mamba_encoder(emb_inputs_r, emb_motion)
        emb_inputs_r = emb_inputs_r[:, :, :512]

        emb_inputs_r = self.LayerNorm(emb_inputs_r)

        if self.output_dims != self.hidden_size:
            h_r = self.output_down_proj(emb_inputs_r)
        else:
            h_r = emb_inputs_r
        h_r = h_r.type(x_r.dtype)
        return h_r
