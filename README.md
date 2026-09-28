# MHTG: multi-group hierarchical taxation game

This repository implements the MHTG economic simulation and Multi-Group Proximal Policy Optimization (MGPPO), alongside three comparison methods—IPPO, MAPPO, and MAA2C. Each group contains one government and a configurable number of households. Governments choose fiscal actions every ten environment steps; households act every step. Inter-group capital mobility can be introduced gradually during training.

The implementation is specific to this multi-group setting; results from other economic simulators should not be treated as results of this repository.

## Repository layout

- `agents/MGPPO`, `agents/IPPO`, `agents/MAPPO`, `agents/MAA2C`: learning algorithms
- `env` and `entities`: economic dynamics, observations, actions, rewards, and termination
- `cfg/default.yaml`: economic and training configuration
- `main.py`: training entry point
- `agents/data/advanced_scfp2022.csv`: optional, user-provided initialization data (not included)

## Requirements and installation

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From the repository root:

~~~bash
uv sync
uv run python main.py --help
~~~

The current uv configuration resolves PyTorch from the CUDA 12.6 package index. Users on other CUDA versions or CPU-only systems may need to change the PyTorch source in `pyproject.toml` before installing.

## Training

Run commands from the repository root, because the current implementation loads its configuration and initialization data using paths relative to that directory.

~~~bash
uv run python main.py --alg MGPPO --n_households 100 --seed 1
~~~

Use `--alg IPPO`, `--alg MAPPO`, or `--alg MAA2C` for a baseline. The `--n_households` argument sets the number of households **per group**. Set the number of government groups at `Environment.Entities[0].entity_args.n` in `cfg/default.yaml`. The default training configuration is long-running; review `Trainer.n_epochs` and the evaluation settings before starting an experiment. The government's decision interval and the capital-mobility curriculum are also configured in `cfg/default.yaml`.

Training runs and TensorBoard files are written under `agents/models/<algorithm>/<households>/runX_n=<groups>/`. Model checkpoints and local experiment outputs are intentionally excluded from version control.

## Evaluation during training

The four reinforcement-learning trainers perform configured periodic evaluations during training and save model checkpoints under their run directories. Separate post-training evaluation, statistics-export, and plotting scripts are not included in this minimal training release.

## Initialization data

The derived initialization dataset is not distributed with this repository. To run data-calibrated experiments, provide a compatible CSV at `agents/data/advanced_scfp2022.csv` containing at least the `WGT`, `EDUC`, and `ASSET` columns. If the file is absent, the code generates random test data; those runs are **not** equivalent to data-calibrated experiments.

## Release status

This is a public source-code snapshot, not a licensed open-source release. No license has been selected, so the repository does not grant permission to reuse or redistribute the code. Experiment results and the derived initialization dataset are not included.
