import copy
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter

from agents.MAPPO.MAPPO_agent import MAPPOAgent
from agents.evaluation_statistics import mean_confidence_interval
from agents.log_path import make_logpath
from agents.MAPPO.policy import BoundedPolicy
from omegaconf import OmegaConf
from contextlib import contextmanager
import random
import json
import hashlib
from pathlib import Path


GOVERNMENT_ACTION_NAMES = (
    "income_tax", "income_progressivity", "asset_tax",
    "asset_progressivity", "transfer", "capital_tax",
)
HOUSEHOLD_ACTION_NAMES = ("saving_rate", "labor_time_share")


def save_args(path, args):
    values = args.items() if hasattr(args, "items") else vars(args).items()
    with open(str(path) + "/setting.txt", "w") as file:
        file.write("------------------ start ------------------\n")
        for name, value in values:
            file.write(f"{name} : {value}\n")
        file.write("------------------- end -------------------")


def log_vector_to_tb(writer, prefix, value, step):
    if value is None:
        return
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    values = np.asarray(value).reshape(-1)
    if values.size == 1:
        writer.add_scalar(prefix, float(values[0]), step)
        return
    for index, item in enumerate(values):
        writer.add_scalar(f"{prefix}/{index}", float(item), step)


class HouseholdData:
    """One transition per environment step for household PPO."""

    def __init__(self, rollout, gamma):
        self.global_obses = torch.stack([item["global_obs"] for item in rollout])
        self.private_obses = torch.stack([item["private_obs"] for item in rollout])
        self.gov_actions = torch.stack([item["gov_action"] for item in rollout])
        self.hou_actions = torch.stack([item["hou_action"] for item in rollout])
        self.mean_house_actions = torch.stack(
            [item["mean_house_action"] for item in rollout]
        )
        self.house_latents = torch.stack([item["house_latent"] for item in rollout])
        self.next_global_obses = torch.stack([item["next_global"] for item in rollout])
        self.next_private_obses = torch.stack([item["next_private"] for item in rollout])
        self.next_gov_actions = torch.stack([item["next_gov_action"] for item in rollout])
        self.house_rewards = torch.stack([item["house_reward"] for item in rollout]).float()
        self.house_log_probs = torch.stack(
            [item["house_log_prob"] for item in rollout]
        )
        self.dones = torch.as_tensor(
            [item["done"] for item in rollout],
            dtype=torch.float32,
            device=self.global_obses.device,
        )
        self.episode_ends = torch.as_tensor(
            [item["episode_end"] for item in rollout],
            dtype=torch.float32,
            device=self.global_obses.device,
        )
        self.discounts = torch.full_like(self.dones, float(gamma))


class GovernmentData:
    """One semi-MDP transition per government decision block."""

    def __init__(self, rollout):
        self.global_obses = torch.stack([item["global_obs"] for item in rollout])
        self.gov_actions = torch.stack([item["gov_action"] for item in rollout])
        self.next_gov_actions = torch.stack(
            [item["next_gov_action"] for item in rollout]
        )
        self.past_gov_actions = torch.stack(
            [item["past_gov_action"] for item in rollout]
        ).squeeze(1)
        self.gov_latents = torch.stack([item["gov_latent"] for item in rollout])
        self.next_global_obses = torch.stack([item["next_global"] for item in rollout])
        self.gov_rewards = torch.stack([item["gov_reward"] for item in rollout]).float()
        self.gov_log_probs = torch.stack([item["gov_log_prob"] for item in rollout])
        self.dones = torch.as_tensor(
            [item["done"] for item in rollout],
            dtype=torch.float32,
            device=self.global_obses.device,
        )
        self.episode_ends = torch.as_tensor(
            [item["episode_end"] for item in rollout],
            dtype=torch.float32,
            device=self.global_obses.device,
        )
        self.discounts = torch.as_tensor(
            [item["discount"] for item in rollout],
            dtype=torch.float32,
            device=self.global_obses.device,
        )


class MAPPO:
    """Independent MAPPO trainer for the asynchronous economic environment."""

    algorithm_name = "MAPPO"

    def source_file_names(self):
        return (
            "env/env_core.py", "entities/household.py",
            "agents/MAPPO/MAPPO.py", "agents/MAPPO/trainer.py",
            "agents/MAPPO/MAPPO_agent.py", "agents/MAPPO/agent.py",
            "agents/MAPPO/policy.py",
        )

    def __init__(self, envs, args, evaluation_only=False):
        self.envs = envs
        self.args = args
        self.evaluation_only = bool(evaluation_only)
        shared_gamma = float(getattr(args, "gamma", 0.95))
        self.government_gamma = float(getattr(args, "government_gamma", shared_gamma))
        self.household_gamma = float(getattr(args, "household_gamma", shared_gamma))
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in (
            self.government_gamma, self.household_gamma
        )):
            raise ValueError("government_gamma and household_gamma must lie in [0, 1]")
        self.eval_env = copy.deepcopy(envs)
        self.agents = []
        self.agent_list = [
            role
            for _ in range(envs.government.n_governments)
            for role in ("government", "household")
        ]

        self.args.gov_obs_dim = envs.government.observation_space.shape[0]
        self.args.gov_action_dim = envs.government.action_space.shape[1]
        self.args.house_obs_dim = envs.households.observation_space.shape[0]
        self.args.house_action_dim = envs.households.action_space.shape[2]
        self.gov_action_interval = int(getattr(args, "gov_action_interval", 10))
        self.warmup_epochs = int(getattr(args, "household_warmup_epochs", 0))
        if self.gov_action_interval != envs.gov_action_interval:
            raise ValueError("Trainer and environment government intervals must match.")
        if self.gov_action_interval < 1:
            raise ValueError("gov_action_interval must be at least 1")
        if min(args.epoch_length, envs.episode_length) % self.gov_action_interval:
            raise ValueError("Time-limit rollouts must end on a government decision boundary.")
        self.min_household_rollout_length = 10
        self.min_government_rollout_length = 2
        self.rollout_episodes_per_update = int(
            getattr(args, "rollout_episodes_per_update", 5)
        )
        if self.rollout_episodes_per_update < 1:
            raise ValueError("rollout_episodes_per_update must be at least 1")
        self.eval_interval = int(getattr(args, "eval_interval_epochs", 10))
        if self.eval_interval < 1:
            raise ValueError("eval_interval_epochs must be at least 1")
        self.training_epoch = 0
        self.best_cumulative_gdp = -float("inf")

        self.model_path = None
        self.metadata_path = None
        self.source_hashes = {}
        if not self.evaluation_only:
            self.model_path, _ = make_logpath(
                algo=self.algorithm_name,
                n=self.args.n_households,
                groups=envs.government.n_governments,
            )
            save_args(self.model_path, self.args)
            OmegaConf.save(OmegaConf.create({"Environment": OmegaConf.to_container(envs.cfg),
                                            "Trainer": OmegaConf.to_container(args)}),
                           self.model_path / "config.yaml")
            self.metadata_path = self.model_path / "checkpoint.json"
            self.source_hashes = {
                name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                for name in self.source_file_names()
            }
            with open(self.model_path / "source_hashes.json", "w", encoding="utf-8") as file:
                json.dump(self.source_hashes, file, indent=2)
        self.create_agents()
        self.tb_writer = (
            None if self.evaluation_only else
            SummaryWriter(log_dir=str(self.model_path) + "/tensorboard")
        )

    def create_agents(self):
        for index, name in enumerate(self.agent_list):
            self.agents.append(MAPPOAgent(self.envs, self.args, index, name))

    def load_actors(self, checkpoint_dir):
        """Load every actor from a training-result/model run directory."""
        checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {checkpoint_dir}")
        actor_paths = [checkpoint_dir / f"{index}.pt" for index in range(len(self.agents))]
        missing = [str(path) for path in actor_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing actor checkpoint(s): " + ", ".join(missing)
            )
        for agent, actor_path in zip(self.agents, actor_paths):
            state_dict = torch.load(
                actor_path, map_location=self.envs.device, weights_only=True
            )
            agent.actor.load_state_dict(state_dict, strict=True)
            agent.actor.eval()
        return actor_paths

    def observation_wrapper(self, global_obs, private_obs):
        if not bool(torch.isfinite(global_obs).all()) or not bool(torch.isfinite(private_obs).all()):
            raise FloatingPointError('Non-finite environment observation before normalization')
        global_obs = global_obs.double().clone()
        private_obs = private_obs.double().clone()
        features_per_government = self.envs.features_per_government
        if global_obs.ndim != 2 or global_obs.shape[1] % features_per_government:
            raise ValueError(
                "Global observation does not match the environment feature blocks."
            )
        for offset in range(0, global_obs.shape[1], features_per_government):
            indices = (offset, offset + 1, offset + 3, offset + 4)
            global_obs[:, indices] = torch.log1p(
                global_obs[:, indices].clamp_min(0.0)
            )
        private_obs[:, :, 1] = torch.log1p(private_obs[:, :, 1].clamp_min(0.0))
        global_obs, private_obs = global_obs.float(), private_obs.float()
        if not bool(torch.isfinite(global_obs).all()) or not bool(torch.isfinite(private_obs).all()):
            raise FloatingPointError('Non-finite normalized observation after float32 conversion')
        return global_obs, private_obs

    def _mean_house_action(self, actions, state_env=None):
        wealth = (self.envs if state_env is None else state_env).households.at_next
        top_actions = []
        bottom_actions = []
        top_count = max(1, int(0.1 * self.args.n_households))
        for group in range(self.envs.government.n_governments):
            indices = torch.argsort(wealth[group], descending=True)
            top_actions.append(actions[group, indices[:top_count]])
            bottom_actions.append(actions[group, indices[int(0.5 * self.args.n_households) :]])
        return torch.cat(
            (torch.stack(top_actions).mean(dim=1), torch.stack(bottom_actions).mean(dim=1)),
            dim=-1,
        )

    def _sample_government_actions(self, global_obs, past_gov_action, deterministic=False):
        actions, logs, latents = [], [], []
        with torch.no_grad():
            for group in range(self.envs.government.n_governments):
                policy = BoundedPolicy(self.agents[group * 2].actor(
                    global_obs[group:group+1], past_gov_action))
                action, latent = policy.sample(deterministic)
                actions.append(action.squeeze(0))
                latents.append(latent.squeeze(0))
                logs.append(policy.log_prob(latent).squeeze(0))
        return torch.stack(actions), torch.stack(logs), torch.stack(latents)

    def _sample_household_actions(self, global_obs, private_obs, gov_action, deterministic=False, state_env=None):
        actions, logs, latents = [], [], []
        with torch.no_grad():
            for group in range(self.envs.government.n_governments):
                policy = BoundedPolicy(self.agents[group * 2 + 1].actor(
                    global_obs[group:group+1], private_obs[group:group+1],
                    gov_action.flatten().unsqueeze(0)))
                action, latent = policy.sample(deterministic)
                actions.append(action.squeeze(0))
                logs.append(policy.log_prob(latent).squeeze(0))
                latents.append(latent.squeeze(0))
        house_action = torch.stack(actions)
        return house_action, self._mean_house_action(house_action, state_env), torch.stack(logs), torch.stack(latents)

    @contextmanager
    def _evaluation_randomness(self):
        numpy_state, python_state = np.random.get_state(), random.getstate()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _evaluate_agent(self, seeds=None):
        with self._evaluation_randomness():
            return self._evaluate_agent_impl(seeds=seeds)

    def evaluate_seeds(self, seeds):
        """Public deterministic evaluation entry point for explicit shock seeds."""
        return self._evaluate_agent(seeds=seeds)

    def _update_network(self, household_rollout, government_rollout,
                        government_update_enabled):
        """Update every group after a frozen-policy multi-episode rollout."""
        losses = [[], [], [], []]
        government_data = (
            GovernmentData(government_rollout)
            if government_update_enabled
            and len(government_rollout) >= self.min_government_rollout_length
            else None
        )
        household_data = (
            HouseholdData(household_rollout, self.household_gamma)
            if len(household_rollout) >= self.min_household_rollout_length
            else None
        )
        for group in range(self.envs.government.n_governments):
            if government_data is not None:
                result = self.agents[group * 2].update(government_data, group * 2)
                for target, source in zip(losses, result):
                    target.extend(source)
            if household_data is not None:
                result = self.agents[group * 2 + 1].update(household_data, group * 2 + 1)
                for target, source in zip(losses, result):
                    target.extend(source)
        return tuple(losses)

    def _evaluate_agent_impl(self, seeds=None):
        evaluation_seeds = (
            list(range(self.args.eval_episodes)) if seeds is None
            else [int(seed) for seed in seeds]
        )
        if not evaluation_seeds:
            raise ValueError("At least one evaluation seed is required")
        totals = {
            "government_reward": [], "household_reward": [], "wealth": [],
            "gdp": [], "income_gini": [], "wealth_gini": [],
            "government_action": [], "household_action": [], "debt_to_gdp": [],
            "consumption": [], "capital": [], "effective_income_tax": [],
            "effective_asset_tax": [], "household_consumption_utility": [],
            "household_labor_cost": [],
            "production_capital": [], "capital_tax": [], "government_transfer": [],
            "asset_interest_rate": [],
        }
        episode_lengths = []
        cumulative_gdp = []
        discounted_government_returns = []
        termination_counts = {
            reason: 0 for reason in self.eval_env.termination_reasons
        }
        device = self.envs.households.at_next.device
        governments = self.envs.government.n_governments

        for seed in evaluation_seeds:
            global_obs, private_obs = self.eval_env.eval_reset(seed=seed)
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
            past_gov_action = torch.zeros(
                (1, governments, self.args.gov_action_dim), device=device
            )
            gov_action = None
            episode_values = {name: [] for name in totals}
            discounted_government_return = torch.zeros(governments, device=device)
            for step in range(self.args.epoch_length):
                if step % self.gov_action_interval == 0:
                    gov_action, _, _ = self._sample_government_actions(
                        global_obs, past_gov_action, deterministic=True
                    )
                    past_gov_action = gov_action.unsqueeze(0)
                house_action, _, _, _ = self._sample_household_actions(
                    global_obs, private_obs, gov_action, deterministic=True, state_env=self.eval_env
                )
                next_global, next_private, gov_reward, house_reward, done = self.eval_env.step(
                    {
                        self.envs.government.name: gov_action,
                        self.envs.households.name: house_action,
                    }
                )
                episode_values["government_reward"].append(torch.as_tensor(gov_reward))
                discounted_government_return += (
                    self.government_gamma ** step
                ) * torch.as_tensor(gov_reward)
                episode_values["household_reward"].append(torch.as_tensor(house_reward).mean(dim=1))
                episode_values["wealth"].append(self.eval_env.households.at_next.mean(dim=1))
                episode_values["gdp"].append(self.eval_env.GDP)
                episode_values["income_gini"].append(self.eval_env.income_gini)
                episode_values["wealth_gini"].append(self.eval_env.wealth_gini)
                episode_values["consumption"].append(self.eval_env.consumption.mean(1))
                episode_values["capital"].append(self.eval_env.Kt_next)
                episode_values["effective_income_tax"].append(
                    self.eval_env.income_tax.sum(1) / self.eval_env.income.sum(1).clamp_min(1e-8))
                episode_values["effective_asset_tax"].append(
                    self.eval_env.asset_tax.sum(1) / self.eval_env.households.at.sum(1).clamp_min(1e-8))
                episode_values["household_consumption_utility"].append(
                    self.eval_env.household_reward_components["consumption_utility"].mean(1))
                episode_values["household_labor_cost"].append(
                    self.eval_env.household_reward_components["labor_cost"].mean(1))
                episode_values["debt_to_gdp"].append(
                    self.eval_env.Bt_next.clamp_min(0) / self.eval_env.GDP.clamp_min(1e-8))
                for key, attribute in {
                    "production_capital": "K_production", "capital_tax": "gov_tax",
                    "government_transfer": "Gt",
                    "asset_interest_rate": "household_asset_interest_rate",
                }.items():
                    episode_values[key].append(getattr(self.eval_env, attribute).clone())
                episode_values["government_action"].append(gov_action)
                episode_values["household_action"].append(house_action.mean(dim=1))
                if done:
                    for reason in self.eval_env.last_termination_reasons:
                        termination_counts[reason] += 1
                    break
                global_obs, private_obs = self.observation_wrapper(next_global, next_private)

            if not episode_values["government_reward"]:
                continue
            episode_lengths.append(len(episode_values["government_reward"]))
            gdp_path = torch.stack(episode_values["gdp"])
            cumulative_gdp.append(gdp_path.sum(dim=0))
            discounted_government_returns.append(discounted_government_return)
            for name, values in episode_values.items():
                totals[name].append(torch.stack(values).mean(dim=0))

        if not episode_lengths:
            return None
        action_seed_samples = {
            "_government_action_seed_samples": torch.stack(
                totals["government_action"]
            ),
            "_household_action_seed_samples": torch.stack(
                totals["household_action"]
            ),
        }
        metrics = {
            name: torch.stack(values).mean(dim=0)
            for name, values in totals.items()
        }
        metrics.update(action_seed_samples)
        metrics["mean_gdp_while_alive"] = metrics.pop("gdp")
        metrics["cumulative_gdp"] = torch.stack(cumulative_gdp).mean(dim=0)
        metrics["discounted_government_return"] = torch.stack(discounted_government_returns).mean(dim=0)
        metrics["episode_length"] = float(np.mean(episode_lengths))
        metrics["survival_years"] = (
            metrics["episode_length"] * self.eval_env.year_per_step
        )
        evaluated_episodes = len(episode_lengths)
        metrics["termination_rates"] = {
            reason: count / evaluated_episodes
            for reason, count in termination_counts.items()
        }
        return metrics

    def _save_actors(self):
        for index, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), str(self.model_path / f"{index}.pt"))

    def _log_evaluation(self, metrics, step):
        names = {
            "government_reward": "government_utility",
            "consumption": "consumption_per_household",
            "capital": "capital",
            "production_capital": "production_capital",
            "capital_tax": "capital_tax",
            "government_transfer": "government_transfer",
            "asset_interest_rate": "household_asset_interest_rate",
            "effective_income_tax": "effective_income_tax",
            "effective_asset_tax": "effective_asset_tax",
            "household_consumption_utility": "household_reward/consumption",
            "household_labor_cost": "household_reward/labor_cost",
            "discounted_government_return": "government_discounted_return",
            "household_reward": "mean_households_utility",
            "wealth": "wealth_per_households",
            "mean_gdp_while_alive": "eval/mean_GDP_while_alive",
            "cumulative_gdp": "eval/cumulative_GDP",
            "debt_to_gdp": "fiscal/debt_to_GDP",
            "income_gini": "income_gini",
            "wealth_gini": "wealth_gini",
        }
        for key, name in names.items():
            log_vector_to_tb(self.tb_writer, name, metrics[key], step)
        _, gov_ci_lower, gov_ci_upper = mean_confidence_interval(
            metrics["_government_action_seed_samples"],
            lower_bound=0.0,
            upper_bound=1.0,
        )
        _, house_ci_lower, house_ci_upper = mean_confidence_interval(
            metrics["_household_action_seed_samples"],
            lower_bound=0.0,
            upper_bound=1.0,
        )
        for agent, (values, lower_values, upper_values) in enumerate(
            zip(metrics["government_action"], gov_ci_lower, gov_ci_upper),
            start=1,
        ):
            for action_name, value, lower, upper in zip(
                GOVERNMENT_ACTION_NAMES, values, lower_values, upper_values
            ):
                tag = (
                    f"eval/action/government_agent_{agent}/"
                    f"{action_name}_normalized"
                )
                self.tb_writer.add_scalar(
                    tag, float(value), step,
                )
                self.tb_writer.add_scalar(
                    f"{tag}/ci95_lower", float(lower), step
                )
                self.tb_writer.add_scalar(
                    f"{tag}/ci95_upper", float(upper), step
                )
        for agent, (values, lower_values, upper_values) in enumerate(
            zip(metrics["household_action"], house_ci_lower, house_ci_upper),
            start=1,
        ):
            for action_name, value, lower, upper in zip(
                HOUSEHOLD_ACTION_NAMES, values, lower_values, upper_values
            ):
                tag = f"eval/action/household_agent_{agent}/{action_name}"
                self.tb_writer.add_scalar(
                    tag, float(value), step,
                )
                self.tb_writer.add_scalar(
                    f"{tag}/ci95_lower", float(lower), step
                )
                self.tb_writer.add_scalar(
                    f"{tag}/ci95_upper", float(upper), step
                )
        self.tb_writer.add_scalar(
            "eval/action/ci_seed_count",
            metrics["_government_action_seed_samples"].shape[0],
            step,
        )
        self.tb_writer.add_scalar("eval/survival_years", metrics["survival_years"], step)
        for reason, rate in metrics["termination_rates"].items():
            self.tb_writer.add_scalar(f"eval/termination_rate/{reason}", rate, step)

    @staticmethod
    def _format_metric(value):
        """Format CPU/GPU tensors compactly for the evaluation console log."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.array2string(np.asarray(value), precision=4, separator=", ")

    def _print_evaluation(self, metrics, epoch, checkpoint_saved):
        """Print the same evaluation indicators that are written to TensorBoard."""
        print(
            f"[{datetime.now()}] {self.algorithm_name} evaluation | "
            f"episode: {epoch + 1}, training_epoch: {self.training_epoch}, "
            f"eval_episodes: {self.args.eval_episodes}, "
            f"episode_length: {metrics['episode_length']:.1f}, "
            f"survival_years: {metrics['survival_years']:.1f}, "
            f"checkpoint_saved: {checkpoint_saved}"
        )
        print(
            "  government_rewards:", self._format_metric(metrics["government_reward"]),
            "| discounted_return:", self._format_metric(metrics["discounted_government_return"]),
            "| household_rewards:", self._format_metric(metrics["household_reward"]),
        )
        print("  mean_wealth:", self._format_metric(metrics["wealth"]))
        print(
            "  cumulative_GDP:", self._format_metric(metrics["cumulative_gdp"]),
            "| mean_GDP_while_alive:", self._format_metric(metrics["mean_gdp_while_alive"]),
            "| income_gini:", self._format_metric(metrics["income_gini"]),
            "| wealth_gini:", self._format_metric(metrics["wealth_gini"]),
        )
        print(
            "  fiscal_diagnostics | debt_to_GDP:",
            self._format_metric(metrics["debt_to_gdp"]),
        )
        for agent, values in enumerate(metrics["government_action"], start=1):
            actions = ", ".join(
                f"{name}={float(value):.4f}"
                for name, value in zip(GOVERNMENT_ACTION_NAMES, values)
            )
            print(f"  government_agent_{agent}_actions | {actions}")
        for agent, values in enumerate(metrics["household_action"], start=1):
            print(
                f"  household_agent_{agent}_actions | "
                f"saving_rate={float(values[0]):.4f}, "
                f"labor_time_share={float(values[1]):.4f}"
            )

    def learn(self):
        device = self.envs.households.at_next.device
        governments = self.envs.government.n_governments
        buffered_household_rollout = []
        buffered_government_rollout = []
        buffered_episodes = 0
        for epoch in range(self.args.n_epochs):
            global_obs, private_obs = self.envs.reset()
            global_obs, private_obs = self.observation_wrapper(global_obs, private_obs)
            past_gov_action = torch.zeros(
                (1, governments, self.args.gov_action_dim), device=device
            )
            household_rollout = []
            government_rollout = []
            block = None
            gov_action = None
            last_gov_reward = torch.zeros(governments, device=device)
            last_house_reward = torch.zeros(
                (governments, self.args.n_households), device=device
            )
            done = False
            step_government_rewards = []

            for step in range(self.args.epoch_length):
                if block is None:
                    gov_action, gov_log_prob, gov_latent = self._sample_government_actions(
                        global_obs, past_gov_action, deterministic=(epoch < self.warmup_epochs)
                    )
                    if government_rollout and not government_rollout[-1]["episode_end"]:
                        government_rollout[-1]["next_gov_action"] = gov_action.detach()
                    block = {
                        "global_obs": global_obs.detach(),
                        "past_gov_action": past_gov_action.detach(),
                        "gov_action": gov_action.detach(),
                        "gov_log_prob": gov_log_prob.detach(),
                        "gov_latent": gov_latent.detach(),
                        "discounted_reward": torch.zeros(governments, device=device),
                        "length": 0,
                    }

                house_action, mean_house_action, house_log_prob, house_latent = self._sample_household_actions(
                    global_obs, private_obs, gov_action
                )
                next_global, next_private, gov_reward, house_reward, done = self.envs.step(
                    {
                        self.envs.government.name: gov_action,
                        self.envs.households.name: house_action,
                    },
                    epoch,
                )
                next_global, next_private = self.observation_wrapper(next_global, next_private)
                household_rollout.append(
                    {
                        "global_obs": global_obs.detach(),
                        "private_obs": private_obs.detach(),
                        "gov_action": gov_action.detach(),
                        "hou_action": house_action.detach(),
                        "mean_house_action": mean_house_action.detach(),
                        "house_log_prob": house_log_prob.detach(),
                        "house_latent": house_latent.detach(),
                        "next_global": next_global.detach(),
                        "next_private": next_private.detach(),
                        "next_gov_action": gov_action.detach(),
                        "house_reward": torch.as_tensor(house_reward).detach(),
                        "done": float(done and self.envs.last_termination_reasons != ("time_limit",)),
                        "episode_end": float(done or step + 1 == self.args.epoch_length),
                    }
                )
                block["discounted_reward"] += (
                    self.government_gamma ** block["length"]
                ) * torch.as_tensor(gov_reward)
                block["length"] += 1
                step_government_rewards.append(torch.as_tensor(gov_reward))
                last_gov_reward = torch.as_tensor(gov_reward)
                last_house_reward = torch.as_tensor(house_reward)

                if done or block["length"] == self.gov_action_interval or step + 1 == self.args.epoch_length:
                    government_rollout.append(
                        {
                            "global_obs": block["global_obs"],
                            "past_gov_action": block["past_gov_action"],
                            "gov_action": block["gov_action"],
                            "next_gov_action": block["gov_action"].detach(),
                            "gov_log_prob": block["gov_log_prob"],
                            "gov_latent": block["gov_latent"],
                            "next_global": next_global.detach(),
                            "length": block["length"],
                            "gov_reward": block["discounted_reward"],
                            "discount": self.government_gamma ** block["length"],
                            "done": float(done and self.envs.last_termination_reasons != ("time_limit",)),
                            "episode_end": float(done or step + 1 == self.args.epoch_length),
                        }
                    )
                    past_gov_action = gov_action.unsqueeze(0)
                    block = None

                if block is None and not household_rollout[-1]["done"]:
                    bootstrap_action, _, _ = self._sample_government_actions(next_global, past_gov_action, deterministic=(epoch < self.warmup_epochs))
                    household_rollout[-1]["next_gov_action"] = bootstrap_action.detach()
                    government_rollout[-1]["next_gov_action"] = bootstrap_action.detach()
                if done:
                    break
                global_obs, private_obs = next_global, next_private

            buffered_household_rollout.extend(household_rollout)
            buffered_government_rollout.extend(government_rollout)
            buffered_episodes += 1
            batch_ready = buffered_episodes == self.rollout_episodes_per_update
            update_performed = False
            losses = ([], [], [], [])
            if batch_ready:
                first_buffered_epoch = epoch - buffered_episodes + 1
                government_update_enabled = first_buffered_epoch >= self.warmup_epochs
                losses = self._update_network(
                    buffered_household_rollout,
                    buffered_government_rollout,
                    government_update_enabled,
                )
                update_performed = any(losses)
                if update_performed:
                    self.training_epoch += 1
                buffered_household_rollout = []
                buffered_government_rollout = []
                buffered_episodes = 0
            training_step = self.training_epoch
            attempted_episode = epoch + 1
            for name, values in zip(
                ("government_actor_loss", "government_critic_loss",
                 "households_actor_loss", "households_critic_loss"),
                losses,
            ):
                if values:
                    self.tb_writer.add_scalar(name, float(np.mean(values)), training_step)
            self.tb_writer.add_scalar("train/episode_length", len(household_rollout), attempted_episode)
            economic_termination = done and self.envs.last_termination_reasons != ("time_limit",)
            self.tb_writer.add_scalar(
                "train/economic_termination", float(economic_termination), attempted_episode
            )
            self.tb_writer.add_scalar(
                "train/capital_mobility_weight", float(self.envs.phi), attempted_episode
            )
            self.tb_writer.add_scalar(
                "train/mean_government_reward", float(torch.stack(step_government_rewards).mean()), attempted_episode
            )
            self.tb_writer.add_scalar(
                "train/mean_household_reward", float(torch.stack([x["house_reward"] for x in household_rollout]).mean()), attempted_episode
            )

            if update_performed and self.training_epoch % self.eval_interval == 0:
                metrics = self._evaluate_agent()
                if metrics is not None:
                    self._log_evaluation(metrics, training_step)
                    cumulative_gdp_score = float(metrics["cumulative_gdp"].mean())
                    checkpoint_saved = False
                    if cumulative_gdp_score > self.best_cumulative_gdp:
                        self.best_cumulative_gdp = cumulative_gdp_score
                        self._save_actors()
                        checkpoint_saved = True
                    if checkpoint_saved:
                        metadata = {
                            "training_epoch": self.training_epoch,
                            "selection_metric": "mean_cumulative_gdp",
                            "selection_value": cumulative_gdp_score,
                            "cumulative_gdp": metrics["cumulative_gdp"].tolist(),
                            "survival_years": metrics["survival_years"],
                            "discounted_government_return":
                                metrics["discounted_government_return"].tolist(),
                        }
                        with open(self.metadata_path, "w", encoding="utf-8") as file:
                            json.dump(metadata, file, indent=2)
                    self._print_evaluation(metrics, epoch, checkpoint_saved)

            self.tb_writer.flush()
            print(
                f"[{datetime.now()}] epoch: {epoch + 1}, "
                f"training_epoch: {self.training_epoch}, "
                f"household_steps: {len(household_rollout)}, "
                f"government_decisions: {len(government_rollout)}, "
                f"buffered_episodes: {buffered_episodes}/{self.rollout_episodes_per_update}, "
                f"update_performed: {update_performed}"
            )

        self.tb_writer.close()

    def test(self):
        """Evaluate the current actors without loading a hard-coded checkpoint."""
        return self._evaluate_agent()
