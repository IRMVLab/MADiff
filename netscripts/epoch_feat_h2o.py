# Runs the H2O training and trajectory evaluation loop.
import datetime
import functools
import logging
import os
import time

import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from diffuseq.step_sample import LossAwareSampler
from diffuseq.utils import dist_util
from netscripts import modelio
from netscripts.epoch_utils import AverageMeters, progress_bar as bar

OBSERVATION_RATIO = 0.6
SNAPSHOT_INTERVAL = 20

intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03,
                'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                'w': 3840, 'h': 2160}

def denormalize(traj, intrinsics=None):

    u = traj[:, 0] * intrinsics['w']
    v = traj[:, 1] * intrinsics['h']
    traj2d = np.stack((u, v), axis=1)


    return traj2d


def traj_affordance_dist(hand_traj, contact_point, future_valid=None, invalid_value=9):
    batch_size = contact_point.shape[0]
    expand_size = int(hand_traj.shape[0] / batch_size)
    contact_point = contact_point.unsqueeze(dim=1).expand(-1, expand_size, 2).reshape(-1, 2)
    dist = torch.sum((hand_traj - contact_point) ** 2, dim=1).reshape(batch_size, -1)
    if future_valid is None:
        sorted_dist, sorted_idx = torch.sort(dist, dim=-1, descending=False)
        return sorted_dist[:, 0]
    else:
        dist = dist.reshape(batch_size, 2, -1)
        future_valid = future_valid > 0
        future_invalid = ~future_valid[:, :, None].expand(dist.shape)
        dist[future_invalid] = invalid_value
        sorted_dist, sorted_idx = torch.sort(dist, dim=-1, descending=False)
        selected_dist = sorted_dist[:, :, 0]
        selected_dist, selected_idx = selected_dist.min(dim=1)
        valid = torch.gather(future_valid, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        selected_dist = selected_dist * valid

    return selected_dist

def compute_fde(pred_traj, gt_traj, valid_traj=None, reduction=True):
    pred_last = pred_traj[:, :, -1, :]
    gt_last = gt_traj[:, :, -1, :]

    valid_loc = (gt_last[:, :, 0] >= 0) & (gt_last[:, :, 1] >= 0)\
                & (gt_last[:, :, 0] < 1) & (gt_last[:, :, 1] < 1)

    error = gt_last - pred_last
    error = error * valid_loc[:, :, None]

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])
        error = error ** 2
        fde = torch.sqrt(error.sum(dim=2)) * valid_traj
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=2)
        fde = error * valid_traj
        if reduction:
            fde = fde.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return fde, valid_traj


def compute_ade(pred_traj, gt_traj, valid_traj=None, reduction=True):
    valid_loc = (gt_traj[:, :, :, 0] >= 0) & (gt_traj[:, :, :, 1] >= 0)\
                 & (gt_traj[:, :, :, 0] < 1) & (gt_traj[:, :, :, 1] < 1)

    error = gt_traj - pred_traj
    error = error * valid_loc[:, :, :, None]

    if torch.is_tensor(error):
        if valid_traj is None:
            valid_traj = torch.ones(pred_traj.shape[0], pred_traj.shape[1])
        error = error ** 2
        ade = torch.sqrt(error.sum(dim=3)).mean(dim=2) * valid_traj
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()
    else:
        if valid_traj is None:
            valid_traj = np.ones((pred_traj.shape[0], pred_traj.shape[1]), dtype=int)
        error = np.linalg.norm(error, axis=3)

        ade = (0.25*error[:,:,0]+0.5*error[:,:,1]+0.75*error[:,:,2]+1*error[:,:,3]) * valid_traj
        if reduction:
            ade = ade.sum() / valid_traj.sum()
            valid_traj = valid_traj.sum()

    return ade, valid_traj


def get_traj_observed(traj_all, num_ratios, mask_o):

    traj_input = traj_all[:, None].repeat(1, num_ratios, 1, 1)
    traj_input = traj_input.reshape(traj_all.shape[0] * num_ratios, traj_all.shape[1], traj_all.shape[2])
    traj_mask = mask_o.unsqueeze(-1)
    traj_input = traj_input * traj_mask
    return traj_input

def get_masks(batch_size, ratios, max_frames, nframes, device):

    num_ratios = ratios.size(1)
    mask_o = torch.zeros((batch_size, num_ratios, max_frames)).to(device, non_blocking=True)
    mask_u = torch.zeros((batch_size, num_ratios, max_frames)).to(device, non_blocking=True)
    last_frames = torch.zeros((batch_size, num_ratios)).long()
    for b in range(batch_size):

        num_full = int(nframes[b])
        num_obs = torch.floor(num_full * ratios[b]).to(torch.long)
        last_frames[b] = num_obs - 1

        for i, n_o in enumerate(num_obs):
            mask_o[b, i, :n_o] = 1
            mask_u[b, i, n_o: num_full] = 1
    return mask_o, mask_u, last_frames

class TrainLoop:
    def __init__(
            self,
            start_epoch = 0,
            epochs = 25,
            loader=None,
            evaluate=False,
            use_cuda=True,
            scheduler=None,
            optimizer=None,
            model_hoi_ori=None,
            model_hoi=None,
            obj_head=None,
            pre_encoder=None,
            motion_encoder=None,
            loc_encoder=None,
            glip_encoder=None,
            model_denoise=None,
            diffusion=None,
            post_encoder=None,
            schedule_sampler=None,
            resume=None,
            save_pred=False,
    ):
        self.pre_encoder = self._wrap_ddp(pre_encoder)
        self.model_denoise = self._wrap_ddp(model_denoise)
        self.post_encoder = self._wrap_ddp(post_encoder)
        self.model_hoi = self._wrap_ddp(model_hoi)
        self.obj_head = self._wrap_ddp(obj_head)
        self.motion_encoder = self._wrap_ddp(motion_encoder)
        self.loc_encoder = self._wrap_ddp(loc_encoder)
        self.glip_encoder = self._wrap_ddp(glip_encoder)

        self.diffusion = diffusion
        self.model_hoi_ori = model_hoi_ori


        self.start_epoch = start_epoch
        self.all_epochs = epochs
        self.evaluate = evaluate
        self.optimizer = optimizer
        self.loader = loader
        self.use_cuda = use_cuda
        self.scheduler = scheduler
        self.schedule_sampler = schedule_sampler

        self.gts_affordance_dict, self.preds_affordance_dict, self.preds_affordance_dict_ori = {}, {}, {}
        self.ade_list = []
        self.ade_ori_list = []
        self.fde_list = []
        self.fde_ori_list = []


        if resume is not None:
            self.start_epoch = modelio.load_checkpoint_by_name(self.pre_encoder, resume_path=resume[0], state_dict_name="pre_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.model_denoise, resume_path=resume[0], state_dict_name="model_denoise_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.post_encoder, resume_path=resume[0], state_dict_name="post_encoder_state_dict", strict=False, device=dist_util.dev())

            self.start_epoch = modelio.load_checkpoint_by_name(self.motion_encoder, resume_path=resume[0], state_dict_name="motion_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.loc_encoder, resume_path=resume[0],
            state_dict_name="loc_encoder_state_dict", strict=False, device=dist_util.dev())
            self.start_epoch = modelio.load_checkpoint_by_name(self.glip_encoder, resume_path=resume[0],
            state_dict_name="glip_encoder_state_dict", strict=False, device=dist_util.dev())

            print("finish loading diffusion model from epoch {}".format(self.start_epoch))


        dist_util.sync_params(self.pre_encoder.parameters())
        dist_util.sync_params(self.model_denoise.parameters())
        dist_util.sync_params(self.post_encoder.parameters())
        dist_util.sync_params(self.model_hoi.parameters())
        dist_util.sync_params(self.motion_encoder.parameters())
        dist_util.sync_params(self.loc_encoder.parameters())
        dist_util.sync_params(self.glip_encoder.parameters())
        dist_util.sync_params(self.obj_head.parameters())

        if self.evaluate:
            self.all_epochs = 1
            self.start_epoch = 0


        self.logger = logging.getLogger('main')
        self.logger.setLevel(level=logging.DEBUG)
        now = datetime.datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(self.project_root, "log")
        self.checkpoint_dir = os.path.join(self.project_root, "diffip_weights")
        self.eval_collect_dir = os.path.join(self.project_root, "collected_pred")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.eval_collect_dir, exist_ok=True)
        if evaluate:
            log_file = os.path.join(log_dir, f"eval_{time_str}.log")
        else:
            log_file = os.path.join(log_dir, f"{time_str}.log")
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


        self.save_pred = save_pred

    @staticmethod
    def _wrap_ddp(module):
        return DDP(
            module.to(dist_util.dev()),
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=True,
        )

    def _set_model_mode(self, train_mode):
        self.model_hoi_ori.eval()
        modules = [
            self.model_hoi,
            self.pre_encoder,
            self.model_denoise,
            self.post_encoder,
            self.motion_encoder,
            self.loc_encoder,
            self.glip_encoder,
            self.obj_head,
        ]
        for module in modules:
            if train_mode:
                module.train()
            else:
                module.eval()


    def _get_loc_features(self, traj, num_ratios, mask):

        traj_input = get_traj_observed(traj, num_ratios, mask)
        loc_feat = self.loc_encoder(traj_input)
        return loc_feat


    def run_loop(self):
        self._set_model_mode(train_mode=not self.evaluate)

        for epoch in range(self.start_epoch, self.all_epochs):
            if not self.evaluate:
                print("Using lr {}".format(self.optimizer.param_groups[0]["lr"]))
                self.logger.info("Using lr {}".format(self.optimizer.param_groups[0]["lr"]))

                self.epoch_pass(
                    phase='train',
                    epoch=epoch,
                    train=True,
                )
            else:
                self.epoch_pass(
                    phase='traj',
                    epoch=epoch,
                    train=False,
                )
    def epoch_pass(self, epoch, phase, train=True):
        time_meters = AverageMeters()


        if train:
            loss_meters = AverageMeters()
        else:
            print(f"evaluate epoch {epoch}")

        end = time.time()
        for batch_idx, sample in enumerate(self.loader):

            if train:
                self.optimizer.zero_grad()
                filename, clip, odometry, nframes, traj_gt, glip_feat_raw, motion_feats_raw, ps_pe = sample

                clip = clip.to(dist_util.dev())
                traj_gt = traj_gt.to(dist_util.dev())
                time_meters.add_loss_value("data_time", time.time() - end)
                ratios = torch.ones((traj_gt.shape[0], 1)) * OBSERVATION_RATIO
                batch_size, max_frames, num_ratios = traj_gt.size(0), traj_gt.size(1), ratios.size(1)

                motion_feats_raw = motion_feats_raw.view(motion_feats_raw.shape[0], motion_feats_raw.shape[1], 3*3)
                motion_feat_encoded = self.motion_encoder(motion_feats_raw)

                glip_feat_raw = glip_feat_raw.to(dist_util.dev())
                glip_feat = self.glip_encoder(glip_feat_raw)

                self.output_size = 2
                traj_all = traj_gt[:, :, :self.output_size]
                mask_o, mask_u, last_obs_frames = get_masks(batch_size, ratios, max_frames, nframes, dist_util.dev())
                mask_o_for_input = mask_o.reshape(batch_size * num_ratios, max_frames)
                mask_u_for_input = mask_u.reshape(batch_size * num_ratios, max_frames)
                mask_ou_for_input = mask_o_for_input + mask_u_for_input
                all_loc_feats = self._get_loc_features(traj_all, num_ratios, mask_ou_for_input)
                assert (torch.sum(mask_u_for_input[0]==1)+torch.sum(mask_o_for_input[0]==1)) == torch.sum(mask_ou_for_input[0]==1)

                right_feat = torch.stack((glip_feat, all_loc_feats), dim=2)
                right_feat = right_feat.view(*right_feat.shape[0:2], -1)
                right_feat_encoded = self.pre_encoder(right_feat)

                t, weights = self.schedule_sampler.sample(glip_feat.shape[0], dist_util.dev())

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model_denoise,
                    self.post_encoder,
                    [right_feat_encoded, right_feat_encoded],
                    t,
                    [mask_ou_for_input, mask_o_for_input, mask_u_for_input],
                    motion_feat_encoded,
                )

                loss_feat_dict = compute_losses()
                loss_feat_level = loss_feat_dict["loss_feat_level"]
                rec_feature_r = loss_feat_dict["rec_feature_r"]

                future_feature = rec_feature_r
                pred_future_traj = self.post_encoder(future_feature)
                pred_future_traj_r = pred_future_traj
                broaded_future_mask = torch.broadcast_to(mask_ou_for_input.unsqueeze(dim=-1), traj_gt.shape)

                vanilla_loss_weight = 0.2
                vanilla_future_traj_r = self.post_encoder(right_feat_encoded.contiguous())
                vanilla_future_traj_r_masked = vanilla_future_traj_r * broaded_future_mask
                traj_gt_masked = traj_gt * broaded_future_mask
                vanilla_r_loss = torch.sum((vanilla_future_traj_r_masked - traj_gt_masked) ** 2, dim=-1)
                vanilla_r_loss = vanilla_r_loss.sum(-1) * vanilla_loss_weight

                pred_future_traj_r_masked = pred_future_traj_r * broaded_future_mask
                traj_gt_masked = traj_gt * broaded_future_mask
                future_traj_r_loss = torch.sum((pred_future_traj_r_masked - traj_gt_masked) ** 2, dim=-1)
                future_traj_r_loss = future_traj_r_loss.sum(-1)


                rec_loss_weight = 1
                losses_r = rec_loss_weight*loss_feat_level['mse_r'] + loss_feat_level['tT_loss_r'] + future_traj_r_loss + vanilla_r_loss
                losses_two_hand = losses_r

                if isinstance(self.schedule_sampler, LossAwareSampler):
                    self.schedule_sampler.update_with_local_losses(
                        t, losses_two_hand.detach()
                    )

                loss = (losses_two_hand * weights).mean()

                model_losses = {
                        "future_traj_r_loss":future_traj_r_loss.mean(),

                        "rec_r_loss":loss_feat_level['mse_r'].mean(),
                        "total_loss":loss,
                }

                loss.backward()
                self.optimizer.step()

                for key, val in model_losses.items():
                    if val is not None:
                        loss_meters.add_loss_value(key, val.detach().cpu().item())

                time_meters.add_loss_value("batch_time", time.time() - end)


                if dist_util.get_rank() == 0:
                    self.logger.info(loss_meters.average_meters["total_loss"].avg)


                    suffix = "Epoch:{epoch} "\
                            "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s "\
                            "| future_traj_r_loss: {future_traj_r_loss:.3f} "\
                            "| rec_r_loss: {rec_r_loss:.3f}"\
                            "| total_loss: {total_loss:.3f} ".format(epoch=epoch, batch=batch_idx + 1, size=len(self.loader),
                                                                    data=time_meters.average_meters["data_time"].val,
                                                                    bt=time_meters.average_meters["batch_time"].avg,
                                                                    future_traj_r_loss=model_losses["future_traj_r_loss"],

                                                                    rec_r_loss=model_losses["rec_r_loss"],
                                                                    total_loss=model_losses["total_loss"],
                                                                    )
                    self.logger.info(suffix)
                    bar(suffix)

                end = time.time()


            else:

                filename, clip, odometry, nframes, traj_gt, glip_feat_raw, motion_feats_raw, ps_pe = sample

                clip = clip.to(dist_util.dev())
                traj_gt = traj_gt.to(dist_util.dev())

                time_meters.add_loss_value("data_time", time.time() - end)

                ratios = torch.ones((traj_gt.shape[0], 1)) * OBSERVATION_RATIO
                batch_size, max_frames, num_ratios = traj_gt.size(0), traj_gt.size(1), ratios.size(1)

                motion_feats_raw = motion_feats_raw.view(motion_feats_raw.shape[0], motion_feats_raw.shape[1], 3*3)
                motion_feat_encoded = self.motion_encoder(motion_feats_raw)

                glip_feat_raw = glip_feat_raw.to(dist_util.dev())
                glip_feat = self.glip_encoder(glip_feat_raw)

                self.output_size = 2
                traj_all = traj_gt[:, :, :self.output_size]
                mask_o, mask_u, last_obs_frames = get_masks(batch_size, ratios, max_frames, nframes, dist_util.dev())

                mask_o_for_input = mask_o.reshape(batch_size * num_ratios, max_frames)
                mask_u_for_input = mask_u.reshape(batch_size * num_ratios, max_frames)

                mask_ou_for_input = mask_o_for_input + mask_u_for_input
                all_loc_feats = self._get_loc_features(traj_all, num_ratios, mask_ou_for_input)

                assert (torch.sum(mask_u_for_input[0]==1)+torch.sum(mask_o_for_input[0]==1)) == torch.sum(mask_ou_for_input[0]==1)

                right_feat = torch.stack((glip_feat, all_loc_feats), dim=2)
                right_feat = right_feat.view(*right_feat.shape[0:2], -1)
                right_feat_encoded = self.pre_encoder(right_feat)

                sample_fn = (
                    self.diffusion.p_sample_loop
                )

                with torch.no_grad():
                    len_observation = last_obs_frames + 1
                    assert len_observation[0] == int(torch.sum(mask_o_for_input[0]==1).item())

                    for sample_idx in range(1):
                        for lo in range(len_observation.shape[0]):
                            nframes_this_b = nframes[lo]
                            nfuture = int(nframes_this_b - len_observation[lo])
                            pseudo_future = torch.zeros((nfuture,512))
                            noise_r = torch.randn_like(pseudo_future).to(dist_util.dev())
                            right_feat_encoded[lo, len_observation[lo]:nframes_this_b, :] = noise_r

                        sample_shape = (right_feat_encoded.shape[0], right_feat_encoded.shape[1], right_feat_encoded.shape[2])

                        model_kwargs = {}
                        print("denoising ...")

                        start_idx = 0
                        samples_r = sample_fn(
                            model_denoise=self.model_denoise,
                            shape=sample_shape,
                            noise=[right_feat_encoded[:,start_idx:,...], right_feat_encoded[:,start_idx:,...]],
                            motion_feat_encoded=motion_feat_encoded[:,start_idx:,...],
                            clip_denoised=False,
                            model_kwargs=model_kwargs,
                            clamp_step=0,
                            clamp_first=True,
                            x_start=[right_feat_encoded, right_feat_encoded],
                            gap=999,
                            device=dist_util.dev(),
                            valid_mask = [mask_ou_for_input, mask_o_for_input, mask_u_for_input],
                        )


                        samples_r = samples_r[-1]

                        pred_future_traj = self.post_encoder(samples_r)

                        for b in range(samples_r.shape[0]):
                            num_full = nframes[b]
                            traj_gt_per = traj_gt[b]
                            traj_gt_per = (traj_gt_per + 1.0) * 0.5
                            samples_r_per = pred_future_traj[b]
                            samples_r_per = (samples_r_per + 1.0) * 0.5

                            start_ = len_observation[b]
                            end_ = num_full
                            traj_gt_per_valid = traj_gt_per[start_:end_].cpu().numpy()
                            samples_r_per_valid = samples_r_per[start_:end_].cpu().numpy()

                            displace_errors = np.sqrt(np.sum((samples_r_per_valid - traj_gt_per_valid)**2, axis=-1))

                            ade = np.mean(displace_errors)
                            self.ade_list.append(ade)

                            final_displace_errors = displace_errors[-1]
                            fde = final_displace_errors
                            self.fde_list.append(fde)

                        print(str(batch_idx) + "/" + str(len(self.loader)) + "ade " + str(np.array(self.ade_list).mean()) + " fde " + str(np.array(self.fde_list).mean()))
                        self.logger.info(str(batch_idx) + " ours "+str(np.array(self.ade_list).mean())+" fde "+ str(np.array(self.fde_list).mean()))

        if train:
            warmup_epochs = 0
            snapshot = SNAPSHOT_INTERVAL
            if (epoch + 1 - warmup_epochs) % snapshot == 0 and (dist_util.get_rank() == 0):
                print("save epoch "+str(epoch+1)+" checkpoint")
                modelio.save_checkpoint(
                {
                    "epoch": epoch + 1,


                    "pre_encoder_state_dict": self.pre_encoder.state_dict(),
                    "model_denoise_state_dict": self.model_denoise.state_dict(),
                    "post_encoder_state_dict": self.post_encoder.state_dict(),

                    "motion_encoder_state_dict": self.motion_encoder.state_dict(),
                    "loc_encoder_state_dict": self.loc_encoder.state_dict(),
                    "glip_encoder_state_dict": self.glip_encoder.state_dict(),

                    "optimizer": self.optimizer.state_dict(),
                },
                checkpoint=self.checkpoint_dir,
                filename = f"checkpoint_{epoch+1}.pth.tar")
                torch.save(self.optimizer.state_dict(), os.path.join(self.checkpoint_dir, "optimizer.pt"))

                return loss_meters
        else:
            val_info = {}
            print("ours mean ", np.array(self.ade_list).mean())

            self.logger.info("ours mean ade "+str(np.array(self.ade_list).mean()))

            self.logger.info("ours mean fde "+str(np.array(self.fde_list).mean()))


            device_id_for_save = int(os.environ['LOCAL_RANK'])
            saved_dir = self.eval_collect_dir
            os.makedirs(saved_dir, exist_ok=True)
            np.save(os.path.join(saved_dir, f"ours_ade{device_id_for_save}.npy"), np.array(self.ade_list))
            if int(os.environ['LOCAL_RANK']) == 0:
                world_size = int(os.environ.get("WORLD_SIZE", "1"))
                while 1:
                    rank_files = [os.path.join(saved_dir, f"ours_ade{i}.npy") for i in range(world_size)]
                    if all(os.path.exists(path) for path in rank_files):
                        ade_mean = []
                        for path in rank_files:
                            ade_mean.extend(np.load(path).tolist())
                        print("ours final ade!!!", np.array(ade_mean).mean())
                        break
                    time.sleep(2)
                self.logger.info("ours mean ade final!!!!"+str(np.array(ade_mean).mean()))


            return val_info
