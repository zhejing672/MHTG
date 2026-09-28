from env.env_core import economic_society
from utils.seeds import set_seeds
import os
import argparse
from importlib import import_module
from importlib.util import find_spec
from omegaconf import OmegaConf

TRAINER_IMPORTS = {
    "MGPPO": ("agents.MGPPO.MGPPO", "MGPPO"),
    "MAPPO": ("agents.MAPPO.MAPPO", "MAPPO"),
    "MAA2C": ("agents.MAA2C.MAA2C", "MAA2C"),
    "IPPO": ("agents.IPPO.IPPO", "IPPO"),
    "calibration": ("agents.calibration", "calibration_agent"),
}


def available_algorithms():
    available = []
    for name, (module_name, _) in TRAINER_IMPORTS.items():
        try:
            if find_spec(module_name) is not None:
                available.append(name)
        except ModuleNotFoundError:
            continue
    return tuple(available)


def parse_args():
    parser = argparse.ArgumentParser()
    algorithms = available_algorithms()
    default_algorithm = "MGPPO" if "MGPPO" in algorithms else None
    parser.add_argument("--config", type=str, default='default')
    parser.add_argument(
        "--alg",
        type=str,
        default=default_algorithm,
        required=default_algorithm is None,
        choices=algorithms,
        help="training algorithm (default: MGPPO when available)",
    )
    parser.add_argument('--device-num', type=int, default=1, help='the number of cuda service num')
    parser.add_argument('--n_households', type=int, default=100, help='the number of households per government group')
    parser.add_argument('--seed', type=int, default=1, help='the random seed')
    parser.add_argument('--hidden_size', type=int, default=None, help='override YAML value')
    parser.add_argument('--q_lr', type=float, default=None, help='override YAML value')
    parser.add_argument('--p_lr', type=float, default=None, help='override YAML value')
    parser.add_argument('--batch_size', type=int, default=None, help='override YAML value')
    parser.add_argument('--update_freq', type=int, default=2, help='[10，20，30]')



    args = parser.parse_args()
    return args

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    args = parse_args()
    print(f"Training algorithm: {args.alg}")
    path = args.config
    yaml_cfg = OmegaConf.load(f'./cfg/{path}.yaml')
    yaml_cfg.Trainer["n_households"] = args.n_households
    yaml_cfg.Environment.Entities[1]["entity_args"].n = args.n_households
    yaml_cfg.seed = args.seed
    for name in ("hidden_size", "q_lr", "p_lr", "batch_size"):
        value = getattr(args, name)
        if value is not None:
            yaml_cfg.Trainer[name] = value

    set_seeds(yaml_cfg.seed, cuda=yaml_cfg.Trainer["cuda"])

    module_name, class_name = TRAINER_IMPORTS[args.alg]
    trainer_class = getattr(import_module(module_name), class_name)
    env = economic_society(yaml_cfg.Environment)
    trainer = trainer_class(env, yaml_cfg.Trainer)
    print("n_households: ", yaml_cfg.Trainer["n_households"])
    trainer.learn()
