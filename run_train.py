# Launches training with the selected dataset backend.


import sys
import os
import argparse
sys.path.append('.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_backend", default="h2o", choices=["h2o", "egopat3d"])
    parser.add_argument("--cuda_devices", default="3")
    parser.add_argument("--extra_args", default="")
    args = parser.parse_args()

    config_by_backend = {
        "h2o": "./configs/h2o.yml",
        "egopat3d": "./configs/egopat3d.yml",
    }
    config_path = config_by_backend[args.dataset_backend]

    COMMANDLINE = (
        f"CUDA_VISIBLE_DEVICES={args.cuda_devices} python traineval.py "
        f"--dataset_backend={args.dataset_backend} --config {config_path} {args.extra_args}"
    )

    print(COMMANDLINE)
    os.system(COMMANDLINE)
