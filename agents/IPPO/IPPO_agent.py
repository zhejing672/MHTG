"""Independent PPO agent using the shared stable on-policy update routine."""
import torch

from agents.MGPPO.MGPPO_agent import MGPPO_agent
from agents.IPPO.policy import (
    GovernmentIndependentValue,
    HouseholdIndependentValue,
    IndependentGovernmentPolicy,
    IndependentHouseholdPolicy,
)


class IPPOAgent(MGPPO_agent):
    """Separate policy/value model for each competitive jurisdiction and role."""

    def __init__(self, envs, args, agent_id, agent_name):
        super().__init__(envs, args, agent_id, agent_name)
        group = agent_id // 2
        self.actor = (
            IndependentHouseholdPolicy(args, group) if self.household else
            IndependentGovernmentPolicy(args, group)
        ).to(envs.device)
        self.critic = (
            HouseholdIndependentValue(args, group) if self.household else
            GovernmentIndependentValue(args)
        ).to(envs.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.p_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.q_lr)


IPPO_agent = IPPOAgent
