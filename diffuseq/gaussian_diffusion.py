# Implements diffusion schedules, losses, and sampling loops.
import math

import numpy as np
import torch as th

from .utils.nn import mean_flat


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):

    if schedule_name == "linear":


        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: 1-np.sqrt(t + 0.0001),
        )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,
        )
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001
        beta_end = scale * 0.02
        first_part = np.linspace(
            beta_start, beta_mid, 10, dtype=np.float64
        )
        second_part = np.linspace(
            beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64
        )
        return np.concatenate(
            [first_part, second_part]
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):

    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class GaussianDiffusion:


    def __init__(
        self,
        *,
        betas,
        predict_xstart,
        rescale_learned_sigmas,
        learn_sigmas,
        sigma_small,
        use_kl,
        rescale_timesteps=False,
    ):
        self.rescale_timesteps = rescale_timesteps
        self.predict_xstart = predict_xstart
        self.rescale_learned_sigmas = rescale_learned_sigmas
        self.learn_sigmas = learn_sigmas
        self.sigma_small = sigma_small
        self.use_kl = use_kl


        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)


        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)


        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )


        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mapping_func = None
        self.add_mask_noise = False

    def training_losses(self, model, post_model, *args, **kwargs):
        self.model = model
        self.post_model = post_model
        return self.training_losses_seq2seq(model, post_model, *args, **kwargs)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):

        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def q_mean_variance(self, x_start, t):

        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None, mask=None):

        if noise is None:
            noise = th.randn_like(x_start)

        assert noise.shape == x_start.shape

        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

        if mask is None:
            return x_t

        mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)
        return th.where(mask == 0, x_start, x_t)

    def q_posterior_mean_variance(self, x_start, x_t, t):

        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, motion_feat_encoded, t, valid_mask, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):

        if model_kwargs is None:
            model_kwargs = {}

        B = x.size(0)
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), motion_feat_encoded, valid_mask, **model_kwargs)


        model_variance = np.append(self.posterior_variance[1], self.betas[1:])
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x, t)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.predict_xstart:
            pred_xstart = process_xstart(model_output)
        else:

            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )


        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None,
            top_p=None, mask=None, x_start=None, motion_feat_encoded=None, valid_mask=None,
            valid_mask_observed=None,
    ):

        out = self.p_mean_variance(
            model,
            x,
            motion_feat_encoded,
            t,
            valid_mask=valid_mask,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=model_kwargs,
        )
        top_p=None
        if top_p is not None and top_p > 0:
            noise = th.randn_like(x)
            replace_mask = th.abs(noise) > top_p
            while replace_mask.any():
                noise[replace_mask] = th.randn_like(noise[replace_mask])
                replace_mask = th.abs(noise) > top_p
            assert (th.abs(noise) <= top_p).all()
        else:
            noise = th.randn_like(x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise


        valid_mask_observed_for_anchor = th.broadcast_to(valid_mask_observed.unsqueeze(dim=-1), x_start.shape)
        sample = th.where(valid_mask_observed_for_anchor == 1, x_start, sample)

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "greedy_mean": out["mean"],
            "out": out
        }


    def p_sample_loop(
        self,
        model_denoise,
        shape,
        noise=None,
        motion_feat_encoded=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
        valid_mask=None,
    ):

        final_r = []

        for sample in self.p_sample_loop_progressive(
            model_denoise,
            shape,
            noise=noise,
            motion_feat_encoded=motion_feat_encoded,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            top_p=top_p,
            clamp_step=clamp_step,
            clamp_first=clamp_first,
            mask=mask,
            x_start=x_start,
            gap=gap,
            valid_mask=valid_mask,
        ):
            final_r.append(sample[0]['sample'])
        return final_r

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        motion_feat_encoded=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=None,
        valid_mask=None,
    ):


        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x_r = noise[0]
        else:
            sample_x_r = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1][::gap]

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            if not clamp_first:
                if i > clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = denoised_fn
            else:
                if i >= clamp_step:
                    denoised_fn_cur = None
                else:
                    denoised_fn_cur = None

            valid_mask_r = valid_mask[0].to(t.device)
            valid_mask_observed = valid_mask[1].to(t.device)
            with th.no_grad():
                out_r = self.p_sample(
                    model,
                    sample_x_r,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn_cur,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start[0],
                    motion_feat_encoded=motion_feat_encoded,
                    valid_mask=valid_mask_r,
                    valid_mask_observed=valid_mask_observed,
                )

                yield [out_r]
                sample_x_r = out_r["sample"]


    def _get_x_start(self, x_start_mean, std):

        noise = th.randn_like(x_start_mean)
        assert noise.shape == x_start_mean.shape
        return (
             x_start_mean + std * noise
        )

    def _token_discrete_loss(self, x_t, get_logits, input_ids, mask=None, truncate=False, t=None):

        reshaped_x_t = x_t
        logits = get_logits(reshaped_x_t)
        loss_fct = th.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        if mask is not None:
            decoder_nll *= mask
        if mask is not None:
            decoder_nll = decoder_nll.sum(dim=-1) / mask.sum(dim=-1)
        else:
            decoder_nll = decoder_nll.mean(dim=-1)

        return decoder_nll

    def _x0_helper(self, model_output, x, t):

        if self.predict_xstart:
            pred_xstart = model_output
            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        else:
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

            pred_prev, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )

        return {'pred_xprev':pred_prev, 'pred_xstart':pred_xstart}


    def training_losses_seq2seq(self, model, post_model, rl_x_start, t, rl_valid_mask, motion_feat_encoded, model_kwargs=None, noise=None):

        x_start_r = rl_x_start[0].to(t.device)
        valid_mask_r = rl_valid_mask[0].to(t.device)


        future_mask = th.ones_like(rl_valid_mask[1]).to(rl_valid_mask[1].device) - rl_valid_mask[1]
        x_start_mean_r = x_start_r

        std = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod,
                                   th.tensor([0]).to(x_start_mean_r.device),
                                   x_start_mean_r.shape)
        x_start_r = self._get_x_start(x_start_mean_r, std)

        noise_r = th.randn_like(x_start_r)
        x_t_r = self.q_sample(x_start_r, t, noise=noise_r, mask=future_mask)

        terms = {}

        target_r = x_start_r
        model_kwargs = {}

        model_output_r = model(x_t_r, self._scale_timesteps(t), motion_feat_encoded, valid_mask_r, **model_kwargs)
        assert model_output_r.shape == target_r.shape == x_start_r.shape


        valid_mask_ou_for_loss = th.broadcast_to(valid_mask_r.unsqueeze(dim=-1), target_r.shape)
        target_r_masked = target_r * valid_mask_ou_for_loss
        model_output_r_masked = model_output_r * valid_mask_ou_for_loss
        loss_recon_r = mean_flat((target_r_masked - model_output_r_masked) ** 2)

        model_out_x_start_r = self._x0_helper(model_output_r, x_t_r, t)['pred_xstart']

        t0_mask = (t == 0)
        x_start_mean_r_masked = x_start_mean_r * valid_mask_ou_for_loss
        model_out_x_start_r_masked = model_out_x_start_r * valid_mask_ou_for_loss
        t0_loss_r = mean_flat((x_start_mean_r_masked - model_out_x_start_r_masked) ** 2)
        loss_recon_r = th.where(t0_mask, t0_loss_r, loss_recon_r)

        terms["mse_r"] = loss_recon_r

        out_mean_r, _, _ = self.q_mean_variance(x_start_r, th.LongTensor([self.num_timesteps - 1]).to(x_start_r.device))

        tT_loss_r = mean_flat(out_mean_r ** 2)
        terms["tT_loss_r"] = tT_loss_r

        return {"loss_feat_level": terms, "rec_feature_r": model_out_x_start_r}

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None,
        valid_mask=None,
        motion_feat_encoded=None,
    ):

        out = self.p_mean_variance(
            model,
            x,
            motion_feat_encoded,
            t,
            valid_mask=valid_mask,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )


        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        if langevin_fn:
            sample = langevin_fn(sample, mean_pred, sigma, self.alphas_cumprod_prev[t[0]], t, x)

        if mask is None:
            pass
        else:

            sample = th.cat((x_start, sample[:, -4:]), dim=1)
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):

        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )


        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)


        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        top_p=None,
        clamp_step=None,
        clamp_first=None,
        mask=None,
        x_start=None,
        gap=1,
    ):

        final = []
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            mask=mask,
            x_start=x_start,
            gap = gap
        ):
            final.append(sample['sample'])
        return final

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        langevin_fn=None,
        mask=None,
        x_start=None,
        gap=1,
    ):

        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            sample_x = noise
        else:
            sample_x = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1][::gap]

        if progress:

            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    sample_x,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    mask=mask,
                    x_start=x_start
                )
                yield out
                sample_x = out["sample"]

def _extract_into_tensor(arr, timesteps, broadcast_shape):

    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def space_timesteps(num_timesteps, section_counts):

    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):


    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, post_model, *args, **kwargs
    ):
        return super().training_losses(self._wrap_model(model), post_model, *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):

        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, motion_feat_encoded, valid_mask, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, motion_feat_encoded, valid_mask, **kwargs)
