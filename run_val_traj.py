# Launches trajectory evaluation with the selected dataset backend.

import sys
import os
import argparse
sys.path.append('.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_backend", default="egopat3d", choices=["h2o", "egopat3d"])
    parser.add_argument("--cuda_devices", default="3")
    args = parser.parse_args()

    resume_by_backend = {
        "h2o": "./diffip_weights/checkpoint_h2o.pth.tar",
        "egopat3d": "./diffip_weights/checkpoint_egopat3d.pth.tar",
    }
    config_by_backend = {
        "h2o": "./configs/h2o.yml",
        "egopat3d": "./configs/egopat3d.yml",
    }
    config_path = config_by_backend[args.dataset_backend]
    resume_path = resume_by_backend[args.dataset_backend]

    COMMANDLINE = (
        f"CUDA_VISIBLE_DEVICES={args.cuda_devices} python traineval.py "
        f"--dataset_backend={args.dataset_backend} --config {config_path} "
        f"--evaluate --resume={resume_path} --traj_only"
    )

    print(COMMANDLINE)
    os.system(COMMANDLINE)
