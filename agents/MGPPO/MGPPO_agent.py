"""On-policy PPO with state-value GAE and proper time-limit bootstrapping."""
import torch
from torch.nn import functional as F

from agents.MGPPO.policy import BoundedPolicy, Policy, HouseholdValue, GovernmentValue


class MGPPO_agent:
    def __init__(self, envs, args, agent_id, agent_name):
        self.args, self.envs = args, envs
        self.agent_id, self.name = agent_id, agent_name
        self.household = agent_name == "household"
        self.uses_current_action_critic = type(self) is MGPPO_agent
        legacy_lambda = float(getattr(args, "lmbda", 0.95))
        lambda_name = "household_lambda" if self.household else "government_lambda"
        self.trace_lambda = float(getattr(args, lambda_name, legacy_lambda))
        if not 0 <= self.trace_lambda <= 1:
            raise ValueError(f"{lambda_name} must lie in [0, 1]")
        governments = envs.government.n_governments
        self.actor = Policy(
            args.house_obs_dim if self.household else args.gov_obs_dim,
            args.gov_action_dim * governments,
            args.house_action_dim if self.household else args.gov_action_dim,
            args, household=self.household,
        ).to(envs.device)
        self.critic = (
            HouseholdValue(args, governments) if self.household else
            GovernmentValue(args, governments, agent_id // 2, envs.features_per_government)
        ).to(envs.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.p_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.q_lr)
        self.value_normalization_epsilon = float(
            getattr(args, "value_normalization_epsilon", 1e-4)
        )
        self.value_huber_delta = float(getattr(args, "value_huber_delta", 1.0))
        if self.value_normalization_epsilon <= 0:
            raise ValueError("value_normalization_epsilon must be positive")
        if self.value_huber_delta <= 0:
            raise ValueError("value_huber_delta must be positive")
        self.value_target_mean = torch.zeros((), device=envs.device)
        self.value_target_variance = torch.ones((), device=envs.device)
        self.value_target_count = 0

    @property
    def value_target_std(self):
        return self.value_target_variance.clamp_min(
            self.value_normalization_epsilon ** 2
        ).sqrt()

    @torch.no_grad()
    def _update_value_normalizer(self, targets):
        flat_targets = targets.detach().reshape(-1)
        if flat_targets.numel() == 0 or not torch.isfinite(flat_targets).all():
            raise FloatingPointError("Invalid critic targets for normalization")
        batch_count = flat_targets.numel()
        batch_mean = flat_targets.mean()
        batch_variance = flat_targets.var(unbiased=False)
        if self.value_target_count == 0:
            self.value_target_mean.copy_(batch_mean)
            self.value_target_variance.copy_(batch_variance)
            self.value_target_count = batch_count
            return
        total_count = self.value_target_count + batch_count
        delta = batch_mean - self.value_target_mean
        combined_mean = self.value_target_mean + delta * batch_count / total_count
        old_sum_squares = self.value_target_variance * self.value_target_count
        batch_sum_squares = batch_variance * batch_count
        combined_sum_squares = (
            old_sum_squares + batch_sum_squares
            + delta.square() * self.value_target_count * batch_count / total_count
        )
        self.value_target_mean.copy_(combined_mean)
        self.value_target_variance.copy_(combined_sum_squares / total_count)
        self.value_target_count = total_count

    def _normalize_value(self, value):
        return (value - self.value_target_mean) / self.value_target_std

    def _denormalize_value(self, normalized_value):
        return normalized_value * self.value_target_std + self.value_target_mean

    def _critic_loss(self, prediction, normalized_target):
        return F.smooth_l1_loss(
            prediction, normalized_target, beta=self.value_huber_delta
        )

    def _returns_and_advantages(self, rewards, values, next_values, discounts,
                                terminals, episode_ends, trace_discounts=None):
        mask = 1 - terminals
        deltas = rewards + discounts * mask * next_values - values
        trace_lambda = float(getattr(self, "trace_lambda",
                                     getattr(self.args, "lmbda", 0.95)))
        trace = discounts * trace_lambda if trace_discounts is None else trace_discounts
        advantages = torch.zeros_like(values)
        acc = torch.zeros_like(values[-1])
        for t in reversed(range(len(rewards))):
            continuation = mask[t] * (1 - episode_ends[t])
            acc = deltas[t] + trace[t] * continuation * acc
            advantages[t] = acc
        targets = advantages + values
        std = advantages.std(unbiased=False)
        advantages = (advantages - advantages.mean()) / std.clamp_min(1e-8)
        if not torch.isfinite(targets).all() or not torch.isfinite(advantages).all():
            raise FloatingPointError("Non-finite PPO targets/advantages")
        return targets, advantages

    def update(self, data, index):
        group = index // 2
        obs = data.global_obses[:, group]
        if self.household:
            private = data.private_obses[:, group]
            context = data.gov_actions.flatten(start_dim=1)
            rewards, old_logs = data.house_rewards[:, group], data.house_log_probs[:, group]
            latent = data.house_latents[:, group]
            if self.uses_current_action_critic:
                value_args = (
                    obs, private, data.hou_actions[:, group],
                    data.mean_house_actions,
                )
                next_args = (
                    data.next_global_obses[:, group],
                    data.next_private_obses[:, group],
                    data.next_hou_actions[:, group],
                    data.next_mean_house_actions,
                )
            else:
                value_args = (obs, private, context)
                next_args = (
                    data.next_global_obses[:, group], data.next_private_obses[:, group],
                    data.next_gov_actions.flatten(start_dim=1),
                )
            discounts = data.discounts[:, None]
            terminals, episode_ends = data.dones[:, None], data.episode_ends[:, None]
            trace = discounts * self.trace_lambda
        else:
            context = data.past_gov_actions
            rewards, old_logs = data.gov_rewards[:, group], data.gov_log_probs[:, group]
            latent = data.gov_latents[:, group]
            if self.uses_current_action_critic:
                value_args = (
                    obs, data.gov_actions, data.mean_house_actions[:, group],
                )
                next_args = (
                    data.next_global_obses[:, group],
                    data.next_gov_actions,
                    data.next_mean_house_actions[:, group],
                )
            else:
                value_args = (obs, context)
                next_args = (data.next_global_obses[:, group], data.gov_actions)
            discounts, terminals = data.discounts, data.dones
            episode_ends = data.episode_ends
            trace = discounts * self.trace_lambda
        with torch.no_grad():
            values = self._denormalize_value(self.critic(*value_args))
            next_values = self._denormalize_value(self.critic(*next_args))
            targets, advantages = self._returns_and_advantages(
                rewards, values, next_values, discounts, terminals, episode_ends, trace)
            if self.uses_current_action_critic:
                policy_returns = targets.detach()
                advantages = (
                    policy_returns - policy_returns.mean()
                ) / policy_returns.std(unbiased=False).clamp_min(1e-8)
            self._update_value_normalizer(targets)
            normalized_targets = self._normalize_value(targets)

        actor_losses, critic_losses = [], []
        steps = int(getattr(self.args, "ppo_minibatch_steps", 64))
        for _ in range(self.args.ppo_epoch):
            for batch in torch.randperm(len(obs), device=obs.device).split(steps):
                params = self.actor(obs[batch], private[batch], context[batch]) if self.household else self.actor(obs[batch], context[batch])
                policy = BoundedPolicy(params)
                log_ratio = policy.log_prob(latent[batch]) - old_logs[batch]
                ratio = log_ratio.clamp(-20, 20).exp()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                if float(approx_kl.detach()) > 1.5 * self.args.max_kl:
                    break
                surrogate = torch.minimum(
                    ratio * advantages[batch],
                    ratio.clamp(1-self.args.eps, 1+self.args.eps) * advantages[batch])
                entropy = policy.entropy().mean()
                actor_loss = -surrogate.mean() - self.args.ent_coef * entropy
                critic_loss = self._critic_loss(
                    self.critic(*(a[batch] for a in value_args)),
                    normalized_targets[batch],
                )
                if not torch.isfinite(actor_loss + critic_loss):
                    raise FloatingPointError("Non-finite PPO loss")
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.max_grad_norm)
                self.critic_optim.step()
                actor_losses.append(float(actor_loss.detach()))
                critic_losses.append(float(critic_loss.detach()))
            else:
                continue
            break
        return ([], [], actor_losses, critic_losses) if self.household else (actor_losses, critic_losses, [], [])
