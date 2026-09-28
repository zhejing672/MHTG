"""Synchronous multi-agent advantage actor-critic update."""
import torch

from agents.MAPPO.MAPPO_agent import MAPPOAgent
from agents.MGPPO.policy import BoundedPolicy


class MAA2CAgent(MAPPOAgent):
    """A2C with decentralized actors and no household action in the government critic."""

    def update(self, data, index):
        group = index // 2
        obs = data.global_obses[:, group]
        if self.household:
            private = data.private_obses[:, group]
            context = data.gov_actions.flatten(start_dim=1)
            rewards = data.house_rewards[:, group]
            latent = data.house_latents[:, group]
            value_args = (obs, private, context)
            next_args = (
                data.next_global_obses[:, group],
                data.next_private_obses[:, group],
                data.next_gov_actions.flatten(start_dim=1),
            )
            discounts = data.discounts[:, None]
            terminals = data.dones[:, None]
            episode_ends = data.episode_ends[:, None]
        else:
            context = data.past_gov_actions
            rewards = data.gov_rewards[:, group]
            latent = data.gov_latents[:, group]
            value_args = (obs, data.gov_actions)
            next_args = (
                data.next_global_obses[:, group],
                data.next_gov_actions,
            )
            discounts = data.discounts
            terminals = data.dones
            episode_ends = data.episode_ends
        trace = discounts * self.trace_lambda

        with torch.no_grad():
            values = self._denormalize_value(self.critic(*value_args))
            next_values = self._denormalize_value(self.critic(*next_args))
            targets, advantages = self._returns_and_advantages(
                rewards,
                values,
                next_values,
                discounts,
                terminals,
                episode_ends,
                trace,
            )
            if not self.household:
                policy_returns = targets.detach()
                advantages = (
                    policy_returns - policy_returns.mean()
                ) / policy_returns.std(unbiased=False).clamp_min(1e-8)
            self._update_value_normalizer(targets)
            normalized_targets = self._normalize_value(targets)

        parameters = (
            self.actor(obs, private) if self.household else
            self.actor(obs, context)
        )
        policy = BoundedPolicy(parameters)
        log_prob = policy.log_prob(latent)
        entropy = policy.entropy().mean()
        actor_loss = -(log_prob * advantages.detach()).mean() - self.args.ent_coef * entropy
        critic_loss = self._critic_loss(self.critic(*value_args), normalized_targets)
        if not torch.isfinite(actor_loss + critic_loss):
            raise FloatingPointError("Non-finite MAA2C loss")

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
        self.actor_optim.step()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
        self.critic_optim.step()

        actor_losses = [float(actor_loss.detach())]
        critic_losses = [float(critic_loss.detach())]
        return (
            ([], [], actor_losses, critic_losses) if self.household else
            (actor_losses, critic_losses, [], [])
        )
