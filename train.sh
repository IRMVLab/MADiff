# Runs distributed H2O training via run_train.py.
TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nproc_per_node=1 --master_port=12234 --use_env run_train.py --dataset_backend=h2o
