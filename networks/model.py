# Wraps the HOI network with loss and inference helpers.
import torch
import torch.nn as nn


class Model(nn.Module):


    def __init__(self, net, lambda_obj=None, lambda_traj=None, lambda_obj_kl=None, lambda_traj_kl=None):
        super(Model, self).__init__()
        self.net = net
        self.lambda_obj = lambda_obj
        self.lambda_obj_kl = lambda_obj_kl
        self.lambda_traj = lambda_traj
        self.lambda_traj_kl = lambda_traj_kl

    @staticmethod
    def _apply_weighted_loss(losses, key, loss_value, weight):

        if weight is not None and loss_value is not None:
            weighted_loss = weight * loss_value.sum()
            losses[key] = weighted_loss.detach().cpu()
            return weighted_loss
        losses[key] = 0.
        return 0

    def forward(self, feat, bbox_feat, valid_mask, future_hands=None, contact_point=None, future_valid=None,
                num_samples=5, pred_len=4):


        if self.training:
            losses = {}
            total_loss = 0
            traj_loss, traj_kl_loss, obj_loss, obj_kl_loss = self.net(feat, bbox_feat, valid_mask, future_hands,
                                                                      contact_point, future_valid)
            total_loss += self._apply_weighted_loss(losses, 'traj_loss', traj_loss, self.lambda_traj)
            total_loss += self._apply_weighted_loss(losses, 'traj_kl_loss', traj_kl_loss, self.lambda_traj_kl)
            total_loss += self._apply_weighted_loss(losses, 'obj_loss', obj_loss, self.lambda_obj)
            total_loss += self._apply_weighted_loss(losses, 'obj_kl_loss', obj_kl_loss, self.lambda_obj_kl)

            if total_loss is not None:
                losses["total_loss"] = total_loss.detach().cpu() if torch.is_tensor(total_loss) else total_loss
            else:
                losses["total_loss"] = 0.
            return total_loss, losses

        future_hands_list = []
        contact_points_list = []
        sentence_feature_output = 0


        for sample_idx in range(10):
            future_hands, contact_point, sentence_feature = self.net.module.inference(
                feat, bbox_feat, valid_mask, future_valid=future_valid, pred_len=pred_len
            )
            future_hands_list.append(future_hands)
            contact_points_list.append(contact_point)

            if sample_idx == 0:
                sentence_feature_output = sentence_feature
            else:
                assert torch.all(sentence_feature_output == sentence_feature)

        contact_points = torch.stack(contact_points_list, dim=0)
        assert len(contact_points.shape) == 3
        contact_points = contact_points.transpose(0, 1)

        future_hands_list = torch.stack(future_hands_list, dim=0)
        future_hands_list = future_hands_list.transpose(0, 1)
        return future_hands_list, contact_points, sentence_feature_output
