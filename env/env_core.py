import copy
import random
from fractions import Fraction

import numpy as np
import torch
from gym.spaces import Box

from entities.government import Government
from entities.household import Household

class economic_society:
    name = "wealth distribution economic society"
    termination_reasons = (
        "wealth_gini",
        "income_gini",
        "non_finite_reward",
        "negative_household_assets",
        "non_positive_capital",
        "time_limit",
    )
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.features_per_government = 7

        for entity_arg in cfg['Entities']:
            if entity_arg['entity_name'] == 'household':
                self.households = Household(entity_arg['entity_args'],cfg['Entities'][0]['entity_args']["n"])
            elif entity_arg['entity_name'] == 'government':
                self.government = Government(entity_arg['entity_args'])

        env_args = cfg['env_core']['env_args']
        self.device = self.households.device
        self.alpha = float(Fraction(str(env_args['alpha'])))

        self.depreciation_rate = env_args['depreciation_rate']
        self.interest_rate = None
        self.household_asset_interest_rate_max = float(env_args.get('household_asset_interest_rate_max', 0.05))
        if not np.isfinite(self.household_asset_interest_rate_max) or not 0 <= self.household_asset_interest_rate_max < 1:
            raise ValueError('household_asset_interest_rate_max must be finite and lie in [0, 1).')
        self.consumption_tax_rate = env_args['consumption_tax_rate']
        reward_args = env_args.get('government_reward', {})
        self.gov_output_floor = float(reward_args.get('output_floor', 1e-8))
        utility = env_args.get('household_reward', {})
        self.hours_reference = float(utility.get('hours_reference', 2000.0))
        self.utility_floor = float(utility.get('consumption_floor', 1e-6))
        failure_penalty = env_args.get('capital_failure_penalty', {})
        self.government_capital_failure_penalty = float(
            failure_penalty.get('government', 0.0)
        )
        self.household_capital_failure_penalty = float(
            failure_penalty.get('household', 0.0)
        )
        if not all(
            np.isfinite(value) and value >= 0
            for value in (
                self.government_capital_failure_penalty,
                self.household_capital_failure_penalty,
            )
        ):
            raise ValueError('capital_failure_penalty values must be finite and non-negative.')
        household_reward_values = (
            self.gov_output_floor, self.hours_reference, self.utility_floor,
            float(self.households.CRRA), float(self.households.IFE),
        )
        if not all(np.isfinite(value) for value in household_reward_values):
            raise ValueError('Reward parameters must be finite.')
        if min(self.gov_output_floor, self.hours_reference, self.utility_floor,
               float(self.households.CRRA)) <= 0:
            raise ValueError("Reward floors, reference hours and theta must be positive.")
        if float(self.households.IFE) <= -1:
            raise ValueError("Household labor curvature gamma must be greater than -1.")
        self.year_per_step = float(env_args.get('year_per_step', 1.0))
        if self.year_per_step <= 0:
            raise ValueError('year_per_step must be positive.')
        self.episode_length = int(env_args.get('episode_length', 300))
        self.termination_check_after_steps = int(
            env_args.get('termination_check_after_steps', 20)
        )
        self.wealth_gini_termination_threshold = float(
            env_args.get('wealth_gini_termination_threshold', 0.9)
        )
        self.income_gini_termination_threshold = float(
            env_args.get('income_gini_termination_threshold', 0.8)
        )
        if self.termination_check_after_steps < 0:
            raise ValueError('termination_check_after_steps must be non-negative.')
        if not 0 <= self.wealth_gini_termination_threshold <= 1:
            raise ValueError('wealth_gini_termination_threshold must lie in [0, 1].')
        if not 0 <= self.income_gini_termination_threshold <= 1:
            raise ValueError('income_gini_termination_threshold must lie in [0, 1].')
        self.step_cnt = 0
        self.last_termination_reasons = ()
        self.hours_max = 365 * 24
        self.gov_action_interval = int(env_args.get("gov_action_interval", 10))
        fiscal = env_args.get("fiscal", {})
        self.income_tax_max = float(fiscal.get("income_tax_max", 0.5))
        self.asset_tax_max = float(fiscal.get("asset_tax_max", 0.5))
        self.capital_tax_max = float(fiscal.get("capital_tax_max", 0.5))
        self.progressivity_max = float(fiscal.get("progressivity_max", 2.0))
        self.income_tax_reference = float(fiscal.get("income_reference", 30000.0))
        self.asset_tax_reference = float(fiscal.get("asset_reference", 100000.0))
        self.transfer_max = float(fiscal.get("transfer_gdp_max", 0.5))
        self.bond_interest_rate = float(fiscal.get("bond_interest_rate", 0.4))
        self.capital_mobility = float(fiscal.get("capital_mobility", 1.0))
        self.capital_mobility_curriculum_epochs = int(
            fiscal.get("capital_mobility_curriculum_epochs", 0)
        )
        self.initial_debt_to_gdp = float(fiscal.get("initial_debt_to_gdp", 0.6))
        if not np.isfinite(self.initial_debt_to_gdp) or self.initial_debt_to_gdp < 0:
            raise ValueError('initial_debt_to_gdp must be finite and non-negative.')
        if not 0 < self.alpha < 1:
            raise ValueError('alpha must lie in (0, 1).')
        if not all(0 <= x < 1 for x in (
            self.income_tax_max, self.asset_tax_max, self.capital_tax_max, self.transfer_max)):
            raise ValueError("Tax/transfer bounds must lie in [0,1).")
        if not 0 <= self.capital_mobility <= 1:
            raise ValueError("capital_mobility must lie in [0,1].")
        if self.capital_mobility_curriculum_epochs < 0:
            raise ValueError("capital_mobility_curriculum_epochs must be non-negative.")
        if min(self.income_tax_reference, self.asset_tax_reference) <= 0:
            raise ValueError("Tax reference scales must be positive.")

        global_obs, private_obs = self.reset()
        self.government.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(global_obs.shape[1],), dtype=np.float32
        )
        self.households.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(global_obs.shape[1] + private_obs.shape[2],), dtype=np.float32
        )

    @property
    def action_spaces(self):
        return {self.households.name: self.households.action_space,
                self.government.name: self.government.action_space}

    @property
    def observation_spaces(self):
        return {self.households.name: self.households.observation_space,
                self.government.name: self.government.observation_space}

    def MarketClear(self):

        self.Lt = torch.sum(self.households.e * self.ht, dim=1).squeeze(-1).clamp_min(1e-8)
        self.WageRate = (1 - self.alpha) * torch.pow(self.K_production/self.Lt, self.alpha)

    def generate_gdp(self):

        gdp = (self.K_production ** self.alpha) * (self.Lt ** (1-self.alpha))
        self.per_household_gdp = gdp / self.households.n_households
        return gdp

    def action_wrapper(self, actions):
        '''
        input: (-1,1)之间
        output: (0, 0.9999)
        '''
        return torch.clip(actions, 0.001, 0.99)

    def workinghours_wrapper(self, ht):
        return self.hours_max * ht

    def capital_mobility_weight(self, epoch=None):
        """Linearly open capital mobility during training; evaluate at full mobility."""
        if epoch is None or self.capital_mobility_curriculum_epochs == 0:
            return self.capital_mobility
        progress = min(max(float(epoch), 0.0) / self.capital_mobility_curriculum_epochs, 1.0)
        return self.capital_mobility * progress

    def step(self, action_dict, epoch=None):
        """Capital mobility -> pre-production levy -> production -> fiscal settlement.

        K_available is company capital located in each jurisdiction. The levy
        reduces THIS period's production input K_production and the company's
        payment reduces public debt without a direct household capital-tax debit.
        """
        self.old_per_gdp = self.per_household_gdp.clone()
        actions = {name: torch.as_tensor(value, device=self.device, dtype=torch.float64)
                   for name, value in action_dict.items()}
        if not all(bool(torch.isfinite(value).all()) for value in actions.values()):
            raise ValueError("Actions must be finite.")
        self.valid_action_dict = actions
        self.households.generate_e_ability()
        self.Kt = self.Kt_next.clone()
        self.Bt = self.Bt_next.clone()
        self.households.at = self.households.at_next.clone()

        gov = self.action_wrapper(actions[self.government.name])
        self.government.tau = gov[:, 0] * self.income_tax_max
        self.government.xi = gov[:, 1] * self.progressivity_max
        self.government.tau_a = gov[:, 2] * self.asset_tax_max
        self.government.xi_a = gov[:, 3] * self.progressivity_max
        self.Gt_prob = gov[:, 4] * self.transfer_max
        self.government.tau_k = gov[:, 5] * self.capital_tax_max
        self.avg_tau_k = (self.government.tau_k * self.Kt).sum() / self.Kt.sum().clamp_min(1e-8)
        self.phi = self.capital_mobility_weight(epoch)
        self.capital_flow = -self.phi * (self.government.tau_k - self.avg_tau_k) * self.Kt
        self.K_available = self.Kt + self.capital_flow
        self.gov_tax = self.government.tau_k * self.K_available
        self.K_production = self.K_available - self.gov_tax
        self.Kt = self.K_available

        house = self.action_wrapper(actions[self.households.name])
        self.saving_p = house[:, :, 0:1]
        self.mean_saving_p = self.saving_p.mean()
        self.workingHours = house[:, :, 1:2]
        self.ht = self.workinghours_wrapper(self.workingHours)
        self.MarketClear()
        self.GDP = self.generate_gdp()

        self._update_household_income()
        self.income_tax, self.asset_tax = self.tax_function(self.income, self.households.at)
        self.post_income = self.income - self.income_tax

        self.requested_Gt = self.Gt_prob * self.GDP
        self.Gt = self.requested_Gt
        subsidy = self.Gt[:, None] / self.households.n_households
        disposable_income = self.post_income + subsidy
        self.consumable_resources = disposable_income
        self.consumption = ((1 - self.saving_p[:, :, 0])
                            * self.consumable_resources.clamp_min(0)
                            / (1+self.consumption_tax_rate))
        consumption_tax = self.consumption * self.consumption_tax_rate
        self.tax_array = self.income_tax + self.asset_tax + consumption_tax
        self.primary_deficit = self.Gt - self.tax_array.sum(1) - self.gov_tax
        self.primary_deficit_ratio = (
            self.primary_deficit.clamp_min(0.0) / self.GDP.clamp_min(1e-8)
        )
        self.Bt_next = ((1+self.bond_interest_rate)*self.Bt + self.Gt
                        - self.tax_array.sum(1) - self.gov_tax)
        self.households.at_next = (
            self.households.at - self.asset_tax
            + self.saving_p[:, :, 0] * self.consumable_resources
        )
        self.debt_capital_adjustment = (
            self.Bt_next - (1+self.interest_rate) * self.Bt
        ).clamp_min(0)
        self.Kt_next = (
            self.households.at_next.sum(1)
            + (1+self.interest_rate) * (self.Kt - self.households.at.sum(1))
            - self.debt_capital_adjustment
        )
        self.disposable_income = disposable_income
        self.wealth_gini = self.gini_coef(self.households.at_next)
        self.income_gini = self.gini_coef(self.disposable_income)
        self.capital_failure = self.Kt_next <= 0
        household_utility = self.utility_function(self.consumption, self.ht)
        household_failure_penalty = (
            self.capital_failure[:, None]
            .expand_as(household_utility)
            .to(household_utility.dtype)
            * self.household_capital_failure_penalty
        )
        self.household_reward_components['capital_failure_penalty'] = (
            household_failure_penalty
        )
        self.households_reward = household_utility - household_failure_penalty
        government_failure_penalty = (
            self.capital_failure.to(self.GDP.dtype)
            * self.government_capital_failure_penalty
        )
        self.government_reward = self.gov_reward() - government_failure_penalty

        terminal_reasons = []
        completed_steps = self.step_cnt + 1
        if completed_steps > self.termination_check_after_steps:
            if bool(torch.any(
                self.wealth_gini > self.wealth_gini_termination_threshold
            )):
                terminal_reasons.append("wealth_gini")
            if bool(torch.any(
                self.income_gini > self.income_gini_termination_threshold
            )):
                terminal_reasons.append("income_gini")
        if not self.reward_is_finite or not bool(torch.isfinite(self.households_reward).all()):
            terminal_reasons.append("non_finite_reward")
        if bool(torch.any(self.households.at_next < 0)):
            terminal_reasons.append("negative_household_assets")
        if bool(torch.any(self.capital_failure)):
            terminal_reasons.append("non_positive_capital")
        self.step_cnt += 1
        self.done = bool(terminal_reasons) or self.step_cnt >= self.episode_length
        self.last_termination_reasons = tuple(terminal_reasons) if terminal_reasons else (
            ("time_limit",) if self.done else ())
        next_global, next_private = self.get_obs()
        return next_global, next_private, self.government_reward, self.households_reward, self.done

    def gov_reward(self):
        '''log(Y_t) multiplied by the mean income/wealth equality factor.'''
        log_gdp = torch.log(self.GDP.clamp_min(self.gov_output_floor))
        combined_gini = ((self.income_gini + self.wealth_gini) / 2).clamp(0.0, 1.0)
        equality_factor = 1.0 - combined_gini
        reward = log_gdp * equality_factor
        self.reward_is_finite = bool(torch.isfinite(reward).all())
        return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


    def reset(self, **custom_cfg):
        self.step_cnt = 0
        self.government.reset()
        self.households.reset()
        self.households.at_next = self.households.at_next.double()
        self.households.at = self.households.at.double()
        self.done, self.last_termination_reasons = False, ()
        self.workingHours = torch.full(
            (self.households.n_governments, self.households.n_households, 1),
            self.hours_reference/self.hours_max, device=self.device, dtype=torch.float64)
        self.ht = self.workinghours_wrapper(self.workingHours)
        self.Lt = (self.ht * self.households.e).sum(1).squeeze(-1)
        total_assets = self.households.at_next.sum(dim=1)
        if not bool(torch.isfinite(total_assets).all()) or bool(torch.any(total_assets <= 0)):
            raise ValueError('Initial total household financial assets must be finite and positive.')
        low, high = torch.zeros_like(total_assets), total_assets.clone()
        for _ in range(64):
            capital = (low + high) / 2
            output = capital.pow(self.alpha) * self.Lt.pow(1-self.alpha)
            below = capital + self.initial_debt_to_gdp * output < total_assets
            low = torch.where(below, capital, low)
            high = torch.where(below, high, capital)
        self.Kt_next = (low + high) / 2
        self.Kt = self.Kt_next.clone()
        self.K_available = self.Kt.clone()
        self.K_production = self.Kt.clone()
        self.phi = self.capital_mobility
        self.GDP = self.generate_gdp()
        self.Bt_next = self.initial_debt_to_gdp * self.GDP
        self.Bt = self.Bt_next.clone()
        self.MarketClear()
        self._update_household_income()
        self.gov_tax = torch.zeros_like(self.Kt)
        self.Gt = torch.zeros_like(self.Kt)
        self.primary_deficit = torch.zeros_like(self.GDP)
        self.primary_deficit_ratio = torch.zeros_like(self.GDP)
        return self.get_obs()

    def _update_household_income(self):
        """Household rates follow local net capital returns within configured bounds."""
        self.interest_rate = self.alpha * self.GDP / self.K_available - self.depreciation_rate
        self.household_asset_interest_rate = self.interest_rate.clamp(0, self.household_asset_interest_rate_max)
        self.labor_income = self.WageRate[:, None] * self.households.e[:, :, 0] * self.ht[:, :, 0]
        self.asset_interest = self.household_asset_interest_rate[:, None] * self.households.at
        self.income = self.labor_income + self.asset_interest

    def set_seed(self, seed):
        """设置全局随机种子，保证 reset 结果可复现"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)


    def eval_reset(self, seed=None, **custom_cfg):
        self.set_seed(seed if seed is not None else custom_cfg.get("seed", 42))
        return self.reset()

    def get_obs(self):
        '''
        v0:
            global state: top 10 mean income, top 10 mean asset, top 10 mean e, bottom 50 mean income, bottom 50 mean asset, bottom 50 mean e, wagerate
            private state: e, a

        新设计：每个 group 的 global_state 都包含所有 group 的宏观状态信息，
        也即 per-group global-state 向量是把各组的 global_obs 直接拼接而成。
        '''
        income = self.income
        wealth = self.households.at_next

        global_obss = []
        private_obss = []
        for i in range(self.government.n_governments):
            sorted_income_index = torch.argsort(income[i], descending=True)

            top10_income_index = sorted_income_index[:int(0.1 * self.households.n_households)]
            bottom50_income_index = sorted_income_index[int(0.5 * self.households.n_households):]
            top10_income = income[i, top10_income_index]
            top10_e = self.households.e[i, top10_income_index, :]
            bot50_income = income[i, bottom50_income_index]
            bot50_e = self.households.e[i, bottom50_income_index, :]

            sorted_wealth_index = torch.argsort(wealth[i], descending=True)
            top10_wealth_index = sorted_wealth_index[:int(0.1 * self.households.n_households)]
            bottom50_wealth_index = sorted_wealth_index[int(0.5 * self.households.n_households):]
            top10_wealth = wealth[i, top10_wealth_index]
            bot50_wealth = wealth[i, bottom50_wealth_index]

            global_obs = torch.stack([
                torch.mean(top10_wealth),
                torch.mean(top10_income),
                torch.mean(top10_e),
                torch.mean(bot50_wealth),
                torch.mean(bot50_income),
                torch.mean(bot50_e),
                self.WageRate[i],
            ])
            private_obs = torch.stack((self.households.e[i, :, 0], wealth[i, :]), -1)
            global_obss.append(global_obs)
            private_obss.append(private_obs)

        global_obss = torch.stack(global_obss, dim=0)
        private_obss = torch.stack(private_obss, dim=0)

        global_obss_all = []
        for i in range(self.government.n_governments):
            global_obss_all.append(torch.cat([global_obss[j] for j in range(self.government.n_governments)], dim=0))
        global_obss_all = torch.stack(global_obss_all, dim=0)

        return global_obss_all, private_obss

    def utility_function(self, c_t, h_t):
        """CRRA consumption utility minus convex labor disutility."""
        consumption = c_t.clamp_min(self.utility_floor)
        theta = float(self.households.CRRA)
        consumption_utility = consumption.log() if abs(theta - 1.0) < 1e-8 else (
            consumption.pow(1 - theta) / (1 - theta)
        )
        labor = (h_t.squeeze(-1) / self.hours_max).clamp_min(0.0)
        gamma = float(self.households.IFE)
        labor_cost = labor.pow(1 + gamma) / (1 + gamma)
        self.household_reward_components = {
            "consumption_utility": consumption_utility,
            "labor_cost": labor_cost,
        }
        return consumption_utility - labor_cost

    def gini_coef(self,wealths):
        """
        计算每列（每个群体）的基尼系数。
        wealths.shape = (num_households, num_groups)
        返回 gini.shape = (num_groups,)
        """
        num_groups = wealths.shape[0]
        ginis = torch.zeros(num_groups, device=wealths.device)

        for i in range(num_groups):
            w = wealths[i,:]
            sorted_w, _ = torch.sort(w)

            sorted_w = torch.cat([torch.tensor([0.0], device=w.device), sorted_w])

            cum_wealths = torch.cumsum(sorted_w, dim=0)
            sum_wealths = cum_wealths[-1]

            if sum_wealths == 0:
                ginis[i] = 0.0
                continue


            xarray = torch.arange(len(cum_wealths), device=w.device) / (len(cum_wealths) - 1)
            yarray = cum_wealths / sum_wealths

            B = torch.trapz(yarray, x=xarray)
            A = 0.5 - B
            ginis[i] = A / (A + B)

        return ginis
    def tax_function(self, income, asset):
        """Dimensionless progression with explicit effective-rate ceilings."""
        def tax(base, rate, progression, ceiling, reference):
            taxable = base.clamp_min(0)
            effective = rate[:, None] + (ceiling-rate[:, None]) * (
                1-torch.exp(-progression[:, None]*taxable/reference))
            return taxable * effective
        return (tax(income, self.government.tau, self.government.xi,
                    self.income_tax_max, self.income_tax_reference),
                tax(asset, self.government.tau_a, self.government.xi_a,
                    self.asset_tax_max, self.asset_tax_reference))
