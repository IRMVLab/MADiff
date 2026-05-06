# Parses YAML and command-line configuration for dataset loading.
import yaml
from pprint import pformat
import os
import torch


class ConfigNode(dict):
    def __init__(self, mapping=None, **kwargs):
        super().__init__()
        self.__dict__ = self
        self.update(mapping or {})
        self.update(kwargs)

    def update(self, mapping=None, **kwargs):
        items = {}
        if mapping:
            items.update(mapping)
        items.update(kwargs)
        for key, value in items.items():
            super().__setitem__(key, self._wrap(value))

    @classmethod
    def _wrap(cls, value):
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value


def to_plain_dict(value):
    if isinstance(value, ConfigNode):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value


def parse_configs(parser):
    parser.add_argument('--config', type=str, default='./configs/h2o.yml',
                        help='The relative path of dataset.')
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='The number of workers to load dataset. Default: 4')
    parser.add_argument('--tag', type=str, default='debug',
                        help='The tag to save model results')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = ConfigNode(yaml.safe_load(f))
    cfg.update(vars(args))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg.update(device=device)
    cfg.MODEL.update(device=device)

    root_path = os.path.dirname(__file__)
    cfg.update(root_path=root_path)

    if cfg.tag == 'debug':
        cfg.TRAIN.epoch = 1
        cfg.DATA.load_all = False

    exp_dir = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config_train.yaml'), 'w') as f:
        f.writelines(pformat(to_plain_dict(cfg)))

    return cfg
