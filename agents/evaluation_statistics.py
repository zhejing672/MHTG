"""Statistical helpers shared by algorithm evaluation loggers."""
from __future__ import annotations

import torch
from scipy.stats import t as student_t


def mean_confidence_interval(
    samples: torch.Tensor,
    confidence: float = 0.95,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the sample mean and two-sided Student-t confidence interval."""
    values = torch.as_tensor(samples).float()
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("samples must contain at least one observation")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    mean = values.mean(dim=0)
    sample_count = values.shape[0]
    if sample_count == 1:
        lower = mean.clone()
        upper = mean.clone()
    else:
        critical = float(
            student_t.ppf((1.0 + confidence) / 2.0, df=sample_count - 1)
        )
        standard_error = values.std(dim=0, unbiased=True) / sample_count ** 0.5
        margin = critical * standard_error
        lower = mean - margin
        upper = mean + margin

    if lower_bound is not None:
        lower = lower.clamp_min(float(lower_bound))
    if upper_bound is not None:
        upper = upper.clamp_max(float(upper_bound))
    return mean, lower, upper
