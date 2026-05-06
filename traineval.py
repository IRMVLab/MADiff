# Builds dataloaders, networks, diffusion components, and runs training or evaluation.

import argparse
import random

import numpy as np
import torch

from netscripts.get_network import get_network, get_my_network
from netscripts.get_optimizer import get_optimizer
from options import netsopts, expopts
from diffuseq.step_sample import create_named_schedule_sampler
from basic_utils import create_model_and_diffusion
from diffuseq.utils import dist_util


def resolve_backend_components(dataset_backend):
    backend = dataset_backend.lower()
    if backend == "egopat3d":
        from netscripts.epoch_feat_egopat import TrainLoop
        from src.EgoPAT3DLoader import build_dataloaders
    else:
        from netscripts.epoch_feat_h2o import TrainLoop
        from src.H2OLoader import build_dataloaders
    return TrainLoop, build_dataloaders


def main(args, parser):

    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    dist_util.setup_dist()
    TrainLoop, build_dataloaders = resolve_backend_components(args.dataset_backend)

    num_frames_input = 10
    num_frames_output = 4

    model_hoi_ori = get_network(args, num_frames_input=num_frames_input,
                                num_frames_output=num_frames_output)

    if args.use_cuda and torch.cuda.is_available():
        print("Using {} GPUs !".format(torch.cuda.device_count()))
        model_hoi_ori.to(dist_util.dev())

    start_epoch = 0
    print("Loaded hoi-forecast2022 model checkpoint from epoch {}, ".format(start_epoch))

    print("freezing hoi-forecast2022 model weights")
    for param_hoi in model_hoi_ori.parameters():
        param_hoi.requires_grad = False

    start_epoch = 0
    model_hoi, obj_head = get_my_network(args, num_frames_input=num_frames_input,
                                         num_frames_output=num_frames_output)
    print("do not freeze hoi-forecast2022 model weights")

    print("******************************************************")
    print("******************************************************")

    model_diff_args = {
        "diffusion_steps": 1000,
        "noise_schedule": "sqrt",
        "learn_sigma": False,
        "timestep_respacing": "",
        "predict_xstart": True,
        "rescale_timesteps": True,
        "sigma_small": False,
        "rescale_learned_sigmas": False,
        "use_kl": False,
    }
    print("diffusion setups\n", model_diff_args)
    pre_encoder, model_denoise, diffusion, post_encoder, motion_encoder, loc_encoder, glip_encoder = create_model_and_diffusion(**model_diff_args)

    print("finish building diffusion model")

    schedule_sampler_name = "lossaware"
    schedule_sampler = create_named_schedule_sampler(schedule_sampler_name, diffusion)
    print("finish building schedule sampler")


    if args.evaluate:
        args.epochs = start_epoch + 1
        traj_val_loader = None
        optimizer=None
        scheduler=None
        from src.config import parse_configs
        cfg = parse_configs(parser)
        test_output = build_dataloaders(cfg, phase='test')
        if isinstance(test_output, tuple):
            test_loader = test_output[0]
        else:
            test_loader = test_output

    else:
        from src.config import parse_configs
        cfg = parse_configs(parser)
        train_loader, traj_val_loader = build_dataloaders(cfg, phase='trainval')
        optimizer, scheduler = get_optimizer(args, pre_encoder=pre_encoder, model_denoise=model_denoise,post_encoder=post_encoder, loc_encoder=loc_encoder, glip_encoder=glip_encoder,
                                              train_loader=train_loader, model_hoi=model_hoi, motion_encoder=motion_encoder, obj_head=obj_head)


    if args.evaluate and args.traj_only:
        loader = test_loader
    else:
        loader = train_loader

    TrainLoop(
            epochs = args.epochs,
            loader=loader,
            evaluate=args.evaluate,
            use_cuda=True,
            optimizer=optimizer,
            scheduler=scheduler,
            model_hoi_ori=model_hoi_ori,
            model_hoi=model_hoi,
            obj_head=obj_head,
            pre_encoder=pre_encoder,
            model_denoise=model_denoise,
            diffusion=diffusion,
            post_encoder=post_encoder,
            motion_encoder=motion_encoder,
            loc_encoder=loc_encoder,
            glip_encoder=glip_encoder,
            schedule_sampler=schedule_sampler,
            resume=args.resume
    ).run_loop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HOI Forecasting")
    netsopts.add_nets_opts(parser)
    netsopts.add_train_opts(parser)
    expopts.add_exp_opts(parser)
    args, _ = parser.parse_known_args()

    if args.traj_only: assert args.evaluate, "evaluate trajectory on validation set must set --evaluate"
    main(args, parser)
    print("All done !")
