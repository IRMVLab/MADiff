# Builds the diffusion model, encoders, and diffusion process.

from diffuseq.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffuseq import gaussian_diffusion as gd
from diffuseq.transformer_model import HOIMamba
from diffuseq.pre_encoder import PreEncoder, MotionEncoder, LocEncoder, GLIPEncoder
from diffuseq.pose_encoder import PostEncoder


def create_model_and_diffusion(
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
):
    pre_encoder =  PreEncoder(input_dims=2*512, output_dims=512, encoder_hidden_dims=64)
    post_encoder =  PostEncoder(input_dims=512, output_dims=2, encoder_hidden_dims1=256, encoder_hidden_dims2=64)
    motion_encoder =  MotionEncoder(input_dims=3*3, output_dims=512, encoder_hidden_dims=64)
    loc_encoder =  LocEncoder(2, hidden_features=256, out_features=512)

    glip_encoder =  GLIPEncoder(input_dims_conv=256, output_dims_conv=512,input_dims=7*12,output_dims=1)


    denoised_model = HOIMamba(d_model=1024, n_layers=6)

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return pre_encoder, denoised_model, diffusion, post_encoder, motion_encoder, loc_encoder, glip_encoder
