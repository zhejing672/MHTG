"""Independent actors and value functions for the IPPO baseline."""
import torch
from torch import nn

from agents.MGPPO.policy import ACTION_HIGH, ACTION_LOW


def network(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, outputs),
    )


def initialize_policy_heads(mean, log_std, initial_action, initial_log_std):
    nn.init.zeros_(mean.weight)
    nn.init.zeros_(mean.bias)
    nn.init.zeros_(log_std.weight)
    nn.init.constant_(log_std.bias, initial_log_std)
    if initial_action is None:
        return
    fraction = (torch.as_tensor(list(initial_action)) - ACTION_LOW) / (
        ACTION_HIGH - ACTION_LOW
    )
    if fraction.shape != mean.bias.shape or not bool(
        ((fraction > 0) & (fraction < 1)).all()
    ):
        raise ValueError("Initial IPPO actions must lie strictly inside policy bounds")
    with torch.no_grad():
        mean.bias.copy_(torch.atanh(2 * fraction - 1))


class IndependentGovernmentPolicy(nn.Module):
    """Government policy conditioned on public state and its own past action."""

    def __init__(self, args, group):
        super().__init__()
        self.group = group
        self.action_dim = args.gov_action_dim
        self.trunk = network(
            args.gov_obs_dim + args.gov_action_dim,
            args.hidden_size,
            args.hidden_size,
        )
        self.mean = nn.Linear(args.hidden_size, args.gov_action_dim)
        self.log_std = nn.Linear(args.hidden_size, args.gov_action_dim)
        self.log_std_min = args.log_std_min
        self.log_std_max = args.log_std_max
        initialize_policy_heads(
            self.mean,
            self.log_std,
            getattr(args, "initial_government_action", None),
            args.initial_policy_log_std,
        )

    def forward(self, global_obs, joint_past_actions):
        joint = joint_past_actions.reshape(
            global_obs.shape[0], -1, self.action_dim
        )
        features = self.trunk(
            torch.cat((global_obs, joint[:, self.group]), dim=-1)
        )
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(
            self.log_std_min, self.log_std_max
        )
        return mean, log_std.exp()


class IndependentHouseholdPolicy(nn.Module):
    """Shared household policy conditioned only on its jurisdiction's policy."""

    def __init__(self, args, group):
        super().__init__()
        self.group = group
        self.action_dim = args.gov_action_dim
        self.trunk = network(
            args.house_obs_dim + args.gov_action_dim,
            args.hidden_size,
            args.hidden_size,
        )
        self.mean = nn.Linear(args.hidden_size, args.house_action_dim)
        self.log_std = nn.Linear(args.hidden_size, args.house_action_dim)
        self.log_std_min = args.log_std_min
        self.log_std_max = args.log_std_max
        initialize_policy_heads(
            self.mean,
            self.log_std,
            getattr(args, "initial_household_action", None),
            args.initial_policy_log_std,
        )

    def forward(self, global_obs, private_obs, joint_government_actions):
        households = private_obs.shape[1]
        joint = joint_government_actions.reshape(
            global_obs.shape[0], -1, self.action_dim
        )
        local_action = joint[:, self.group]
        inputs = torch.cat(
            (
                global_obs[:, None].expand(-1, households, -1),
                private_obs,
                local_action[:, None].expand(-1, households, -1),
            ),
            dim=-1,
        )
        features = self.trunk(inputs)
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(
            self.log_std_min, self.log_std_max
        )
        return mean, log_std.exp()


class GovernmentIndependentValue(nn.Module):
    """Independent V_i(o_i); joint actions are intentionally ignored."""

    def __init__(self, args):
        super().__init__()
        self.net = network(args.gov_obs_dim, args.hidden_size, 1)

    def forward(self, global_obs, _joint_past_actions):
        return self.net(global_obs).squeeze(-1)


class HouseholdIndependentValue(nn.Module):
    """Independent household value using only its local fiscal action."""

    def __init__(self, args, group):
        super().__init__()
        self.group = group
        self.action_dim = args.gov_action_dim
        self.net = network(
            args.house_obs_dim + args.gov_action_dim,
            args.hidden_size,
            1,
        )

    def forward(self, global_obs, private_obs, joint_government_actions):
        households = private_obs.shape[1]
        joint = joint_government_actions.reshape(
            global_obs.shape[0], -1, self.action_dim
        )
        local_action = joint[:, self.group]
        inputs = torch.cat(
            (
                global_obs[:, None].expand(-1, households, -1),
                private_obs,
                local_action[:, None].expand(-1, households, -1),
            ),
            dim=-1,
        )
        return self.net(inputs).squeeze(-1)
