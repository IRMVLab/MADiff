# Defines CVAE modules for hand trajectory prediction.
import torch
import torch.nn as nn
from networks.affordance_decoder import VAE


class TrajCVAE(nn.Module):


    def __init__(self, in_dim, hidden_dim, latent_dim, condition_dim, coord_dim=None,
                 condition_contact=False, z_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_contact = condition_contact
        self.z_scale = z_scale
        if self.condition_contact:
            if coord_dim is None:
                coord_dim = hidden_dim // 2
            self.coord_dim = coord_dim
            self.contact_to_feature = nn.Sequential(
                nn.Linear(2, coord_dim, bias=False),
                nn.ELU(inplace=True))
            self.contact_context_fusion = nn.Sequential(
                nn.Linear(condition_dim+coord_dim, condition_dim, bias=False),
                nn.ELU(inplace=True))

        self.cvae = VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=condition_dim)

    def _build_condition_context(self, context, contact_point=None, hand_factor=2):

        if not self.condition_contact:
            return context

        assert contact_point is not None, "contact_point is required when condition_contact is enabled."
        batch_size = contact_point.shape[0]
        time_steps = int(context.shape[0] / batch_size / hand_factor)
        contact_feat = self.contact_to_feature(contact_point)
        contact_feat = contact_feat[:, None, None, :].repeat(1, hand_factor, time_steps, 1)
        contact_feat = contact_feat.reshape(-1, self.coord_dim)
        fusion_feat = torch.cat([context, contact_feat], dim=1)
        return self.contact_context_fusion(fusion_feat)

    def forward(self, context, target_hand, future_valid, contact_point=None, return_pred=False):

        batch_size = future_valid.shape[0]
        condition_context = self._build_condition_context(context, contact_point=contact_point, hand_factor=2)

        if not return_pred:
            recon_loss, KLD = self.cvae(target_hand, c=condition_context)
        else:
            pred_hand, recon_loss, KLD = self.cvae(target_hand, c=condition_context, return_pred=return_pred)

        KLD = KLD.reshape(batch_size, 2, -1).sum(-1)
        KLD = (KLD * future_valid).sum(1)
        recon_loss = recon_loss.reshape(batch_size, 2, -1).sum(-1)
        traj_loss = (recon_loss * future_valid).sum(1)

        if not return_pred:
            return traj_loss, KLD
        return pred_hand, traj_loss, KLD

    def inference(self, context, contact_point=None):

        if self.condition_contact:
            assert contact_point is not None, "contact_point is required when condition_contact is enabled."
            batch_size = contact_point.shape[0]
            time_steps = int(context.shape[0] / batch_size)
            contact_feat = self.contact_to_feature(contact_point)
            contact_feat = contact_feat[:, None, :].repeat(1, time_steps, 1)
            contact_feat = contact_feat.reshape(-1, self.coord_dim)
            fusion_feat = torch.cat([context, contact_feat], dim=1)
            condition_context = self.contact_context_fusion(fusion_feat)
        else:
            condition_context = context

        z = self.z_scale * torch.randn([context.shape[0], self.latent_dim], device=context.device)
        recon_x = self.cvae.inference(z, c=condition_context)
        return recon_x
