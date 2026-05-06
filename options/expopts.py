# Registers experiment command-line options.
def add_exp_opts(parser):
    parser.add_argument("--resume", type=str, nargs="+", metavar="PATH",
                        help="path to latest checkpoint (default: none)")
    parser.add_argument("--evaluate", dest="evaluate", action="store_true",
                        help="evaluate model on validation set")
    parser.add_argument("--use_cuda", default=1, type=int, help="use GPU (default: True)")
    parser.add_argument("--traj_only", action="store_true", help="evaluate traj on validation dataset")
    parser.add_argument(
        "--dataset_backend",
        default="h2o",
        choices=["h2o", "egopat3d"],
        help="select dataloader and train loop backend",
    )
