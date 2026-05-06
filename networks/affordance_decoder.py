# Defines VAE modules for object affordance prediction.
import torch
import torch.nn as nn
from networks.decoder_modules import VAE


class AffordanceCVAE(nn.Module):


    def __init__(self, in_dim, hidden_dim, latent_dim, condition_dim, coord_dim=None,
                 pred_len=4, condition_traj=True, z_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_traj = True
        self.z_scale = z_scale
        if self.condition_traj:
            if coord_dim is None:
                coord_dim = hidden_dim // 2
            self.coord_dim = coord_dim
            self.traj_to_feature = nn.Sequential(
                nn.Linear(2 * (pred_len + 1), coord_dim * (pred_len + 1), bias=False),
                nn.ELU(inplace=True))
            self.traj_context_fusion = nn.Sequential(
                nn.Linear(condition_dim + coord_dim * (pred_len + 1), condition_dim, bias=False),
                nn.ELU(inplace=True))

        self.cvae = VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=condition_dim)

    def _build_condition_context(self, context, hand_traj):

        if not self.condition_traj:
            return context

        assert hand_traj is not None, "hand_traj is required when condition_traj is enabled."
        batch_size = context.shape[0]
        hand_traj = hand_traj.reshape(batch_size, -1)
        traj_feat = self.traj_to_feature(hand_traj)
        fusion_feat = torch.cat([context, traj_feat], dim=1)
        return self.traj_context_fusion(fusion_feat)

    def forward(self, context, contact_point, hand_traj=None, return_pred=False):

        condition_context = self._build_condition_context(context, hand_traj)

        if not return_pred:
            recon_loss, KLD = self.cvae(contact_point, c=condition_context)
            return recon_loss, KLD
        pred_contact, recon_loss, KLD = self.cvae(
            contact_point, c=condition_context, return_pred=return_pred
        )
        return pred_contact, recon_loss, KLD

    def inference(self, context, hand_traj=None):

        condition_context = self._build_condition_context(context, hand_traj)
        z = self.z_scale * torch.randn(
            [condition_context.shape[0], self.latent_dim], device=condition_context.device
        )
        recon_x = self.cvae.inference(z, c=condition_context)
        return recon_x
