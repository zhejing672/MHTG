from pathlib import Path
import re


RUN_DIRECTORY_PATTERN = re.compile(r"^run(\d+)(?:_n=\d+)?$")


def _next_run_name(model_dir, groups):
    """Return a unique run name while accepting both old and new directories."""
    groups = int(groups)
    if groups < 1:
        raise ValueError("groups must be at least 1")
    run_numbers = []
    if model_dir.exists():
        for folder in model_dir.iterdir():
            match = RUN_DIRECTORY_PATTERN.fullmatch(folder.name)
            if folder.is_dir() and match:
                run_numbers.append(int(match.group(1)))
    next_number = max(run_numbers, default=0) + 1
    return f"run{next_number}_n={groups}"


def make_logpath(algo, n, groups, test=False):
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "models" / algo / str(n)
    log_dir = base_dir / "models" / "config_training"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir = model_dir / _next_run_name(model_dir, groups)
    run_dir.mkdir(parents=True)

    return run_dir, log_dir
