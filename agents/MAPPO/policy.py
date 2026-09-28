"""Independent actor and centralized-value networks for MAPPO and MAA2C."""
import math

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


ACTION_LOW = 0.001
ACTION_HIGH = 0.99


def network(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, outputs),
    )


class BoundedPolicy:
    """Affine-tanh policy used consistently for actions and log probabilities."""

    def __init__(self, parameters):
        self.mean, self.std = parameters
        self.normal = Normal(self.mean, self.std)
        self.scale = (ACTION_HIGH - ACTION_LOW) / 2

    def action(self, latent):
        return ACTION_LOW + self.scale * (torch.tanh(latent) + 1)

    def sample(self, deterministic=False):
        latent = self.mean if deterministic else self.normal.rsample()
        return self.action(latent), latent

    def log_prob(self, latent):
        log_jacobian = math.log(self.scale) + 2 * (
            math.log(2) - latent - F.softplus(-2 * latent)
        )
        return (self.normal.log_prob(latent) - log_jacobian).sum(-1)

    def entropy(self):
        return -self.log_prob(self.normal.rsample())


class Policy(nn.Module):
    """Government actor conditioned on public state and joint fiscal history."""

    def __init__(self, obs_dim, context_dim, action_dim, args, household=False):
        super().__init__()
        self.household = household
        self.trunk = network(obs_dim + context_dim, args.hidden_size, args.hidden_size)
        self.mean = nn.Linear(args.hidden_size, action_dim)
        self.log_std = nn.Linear(args.hidden_size, action_dim)
        self.log_std_min, self.log_std_max = args.log_std_min, args.log_std_max
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, args.initial_policy_log_std)
        initial = getattr(
            args,
            "initial_household_action" if household else "initial_government_action",
            None,
        )
        if initial is not None:
            fraction = (
                torch.tensor(list(initial), dtype=self.mean.bias.dtype)
                - ACTION_LOW
            ) / (ACTION_HIGH - ACTION_LOW)
            if fraction.shape != (action_dim,) or not bool(
                ((fraction > 0) & (fraction < 1)).all()
            ):
                raise ValueError(
                    "Initial policy actions must lie strictly inside the policy bounds."
                )
            with torch.no_grad():
                self.mean.bias.copy_(torch.atanh(2 * fraction - 1))

    def forward(self, global_obs, context):
        inputs = torch.cat((global_obs, context.flatten(start_dim=1)), dim=-1)
        features = self.trunk(inputs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(
            self.log_std_min, self.log_std_max
        )
        return mean, log_std.exp()


class HouseholdPolicy(Policy):
    """Household actor that does not observe government actions.

    The optional third argument preserves compatibility with the common trainer
    call signature, but is intentionally ignored.
    """

    def __init__(self, args):
        super().__init__(
            obs_dim=args.house_obs_dim,
            context_dim=0,
            action_dim=args.house_action_dim,
            args=args,
            household=True,
        )

    def forward(self, global_obs, private_obs, _government_actions=None):
        households = private_obs.shape[1]
        inputs = torch.cat(
            (global_obs[:, None].expand(-1, households, -1), private_obs),
            dim=-1,
        )
        features = self.trunk(inputs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(
            self.log_std_min, self.log_std_max
        )
        return mean, log_std.exp()


class GovernmentCentralValue(nn.Module):
    """Central government action-value input, excluding household actions."""

    def __init__(self, args, governments):
        super().__init__()
        self.net = network(
            args.gov_obs_dim + governments * args.gov_action_dim,
            args.hidden_size,
            1,
        )

    def forward(self, global_obs, government_actions):
        inputs = torch.cat(
            (global_obs, government_actions.flatten(start_dim=1)), dim=-1
        )
        return self.net(inputs).squeeze(-1)


class HouseholdCentralValue(nn.Module):
    """V_i(s, private household state, current joint fiscal policy)."""

    def __init__(self, args, governments):
        super().__init__()
        self.net = network(
            args.house_obs_dim + governments * args.gov_action_dim,
            args.hidden_size,
            1,
        )

    def forward(self, global_obs, private_obs, government_actions):
        households = private_obs.shape[1]
        joint_actions = government_actions.flatten(start_dim=1)
        inputs = torch.cat(
            (
                global_obs[:, None].expand(-1, households, -1),
                private_obs,
                joint_actions[:, None].expand(-1, households, -1),
            ),
            dim=-1,
        )
        return self.net(inputs).squeeze(-1)
