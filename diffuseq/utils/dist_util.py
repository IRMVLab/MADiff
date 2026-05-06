# Configures distributed devices and synchronizes parameters.
import os
import socket

import torch as th
import torch.distributed as dist

def setup_dist():

    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())

    if os.environ.get("LOCAL_RANK") is None:
        os.environ["MASTER_ADDR"] = hostname
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        port = _find_free_port()
        os.environ["MASTER_PORT"] = str(port)
        os.environ['LOCAL_RANK'] = str(0)

    dist.init_process_group(backend=backend, init_method="env://")

    if th.cuda.is_available():
        th.cuda.set_device(dev())
        th.cuda.empty_cache()


def dev():

    if th.cuda.is_available():
        return th.device(f"cuda:{os.environ['LOCAL_RANK']}")
    return th.device("cpu")


def sync_params(params):

    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def get_rank():
    return  dist.get_rank()
