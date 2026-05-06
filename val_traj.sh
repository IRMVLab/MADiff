# Runs distributed H2O trajectory evaluation via run_val_traj.py.
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12273 --use_env run_val_traj.py --dataset_backend=h2o
