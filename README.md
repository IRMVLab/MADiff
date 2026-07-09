# [T-PAMI'25] MADiff: Motion-Aware Mamba Diffusion Models for Hand Trajectory Prediction on Egocentric Videos

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2409.02638-b31b1b)](https://arxiv.org/abs/2409.02638)
[![Project](https://img.shields.io/badge/Project-Page-blue)](https://irmvlab.github.io/madiff.github.io/)
[![Code](https://img.shields.io/badge/GitHub-IRMVLab%2FMADiff-black)](https://github.com/IRMVLab/MADiff)
[![3D Extension](https://img.shields.io/badge/3D%20Extension-MMTwin-green)](https://github.com/IRMVLab/MMTwin)

This repository trains and evaluates MADiff-style hand trajectory forecasting models on H2O or EgoPAT3D-style preprocessed data.

Thanks to Haoran Yang for helping organize the code.

## Environment

Use Python 3.8+ and install the runtime packages with:

```bash
pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install the matching PyTorch wheel first, then run the command above for the remaining packages.

## Data And Checkpoints

The default scripts run the H2O backend.

- H2O config: `configs/h2o.yml`
- EgoPAT3D config: `configs/egopat3d.yml`
- H2O default data root: `/data`
- EgoPAT3D default data root: `/data`
- Download the H2O-PT and EgoPAT3D-DT datasets from the dataset instructions in [oppo-us-research/USST](https://github.com/oppo-us-research/USST).
- Download the preprocessed files and MADiff pretrained weights from [SJTU Pan](https://pan.sjtu.edu.cn/web/share/593c95024bbff82cd8d1547157382328).
- Evaluation checkpoints expected by `run_val_traj.py`:
  - H2O: `./diffip_weights/checkpoint_h2o.pth.tar`
  - EgoPAT3D: `./diffip_weights/checkpoint_egopat3d.pth.tar`

With the default configs, H2O data is read from `/data/h2o_dataset`, and EgoPAT3D data is read from `/data/EgoPAT3D-postproc`.

After downloading the pretrained weights, place or rename them to the checkpoint paths listed above.

Edit the YAML config files or pass `--extra_args` through `run_train.py` if your data or checkpoint paths differ.

## Train

```bash
bash train.sh
```

Equivalent direct command:

```bash
python run_train.py --dataset_backend h2o
```

Train EgoPAT3D instead:

```bash
python run_train.py --dataset_backend egopat3d
```

## Evaluate Trajectory

```bash
bash val_traj.sh
```

Equivalent direct command:

```bash
python run_val_traj.py --dataset_backend h2o
```

Evaluate EgoPAT3D instead:

```bash
python run_val_traj.py --dataset_backend egopat3d
```

## Useful Overrides

Select GPU ids used by the wrapper:

```bash
python run_train.py --cuda_devices 0 --dataset_backend h2o
```

Forward extra training arguments to `traineval.py`:

```bash
python run_train.py --dataset_backend h2o --extra_args "--epochs 100 --lr 0.0001"
```

Show available options:

```bash
python traineval.py --help
```

## MADiff for Robotics

<img src="./madiff_for_act_compress.gif" alt="MADiff for ACT" width="50%">

MADiff can improve the performance of robot imitation learning. Feel free to try integrating the predicted motion features into ACT!

## Citation

If this work is useful for your work, kindly cite our paper:

```bibtex
@article{ma2024madiff,
  title={MADiff: Motion-Aware Mamba Diffusion Models for Hand Trajectory Prediction on Egocentric Videos},
  author={Ma, Junyi and Chen, Xieyuanli and Bao, Wentao and Xu, Jingyi and Wang, Hesheng},
  journal={arXiv preprint arXiv:2409.02638},
  year={2024}
}
```

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.
