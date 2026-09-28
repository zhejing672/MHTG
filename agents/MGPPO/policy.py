"""MGPPO policies and state-value baselines (independent of legacy agents)."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

ACTION_LOW = 0.001
ACTION_HIGH = 0.99


class BoundedPolicy:
    """One affine-tanh transform for sampling, execution and likelihoods."""

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


def network(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, outputs))


class Policy(nn.Module):
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
        initial = getattr(args, "initial_household_action" if household else "initial_government_action", None)
        if initial is not None:
            fraction = (torch.tensor(list(initial)) - ACTION_LOW) / (ACTION_HIGH - ACTION_LOW)
            if fraction.shape != (action_dim,) or not bool(((fraction > 0) & (fraction < 1)).all()):
                raise ValueError("Initial policy actions must lie strictly inside the policy bounds.")
            with torch.no_grad():
                self.mean.bias.copy_(torch.atanh(2 * fraction - 1))

    def forward(self, global_obs, context, household_context=None):
        if self.household:
            private_obs = context
            context = household_context.flatten(start_dim=1)
            n = private_obs.shape[1]
            inputs = torch.cat((global_obs[:, None].expand(-1, n, -1),
                                private_obs, context[:, None].expand(-1, n, -1)), -1)
        else:
            inputs = torch.cat((global_obs, context.flatten(start_dim=1)), -1)
        features = self.trunk(inputs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std.exp()


class HouseholdValue(nn.Module):
    """Q from own current action and every group's mean household action."""
    def __init__(self, args, governments):
        super().__init__()
        self.net = network(args.house_obs_dim + args.house_action_dim
                           + governments * args.house_action_dim,
                           args.hidden_size, 1)

    def forward(self, global_obs, private_obs, house_action, mean_house_actions):
        n = private_obs.shape[1]
        joint_mean = mean_house_actions.flatten(start_dim=1)
        inputs = torch.cat((global_obs[:, None].expand(-1, n, -1), private_obs,
                            house_action,
                            joint_mean[:, None].expand(-1, n, -1)), -1)
        return self.net(inputs).squeeze(-1)


class GovernmentValue(nn.Module):
    """Local household-response value plus joint inter-government value."""
    def __init__(self, args, governments, group, features):
        super().__init__()
        self.group, self.features = group, features
        self.action_dim = args.gov_action_dim
        self.weight = float(args.inter_group_weight)
        self.local = network(features + args.gov_action_dim + args.house_action_dim,
                             args.hidden_size, 1)
        self.joint = network(args.gov_obs_dim + governments * args.gov_action_dim,
                             args.hidden_size, 1)

    def forward(self, global_obs, current_actions, mean_house_action):
        current = current_actions.reshape(global_obs.shape[0], -1, self.action_dim)
        start = self.group * self.features
        local = torch.cat((global_obs[:, start:start + self.features],
                           current[:, self.group], mean_house_action), -1)
        joint = torch.cat((global_obs, current.flatten(start_dim=1)), -1)
        return (self.local(local) + self.weight * self.joint(joint)).squeeze(-1)
