from entities.base import BaseEntity
import numpy as np
import pandas as pd
import torch
from gym.spaces import Box


class Household(BaseEntity):
    name = 'Household'

    def __init__(self, entity_args, n_government):
        super().__init__()
        self.n_households = entity_args['n']
        self.n_governments = n_government
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.CRRA = entity_args['CRRA']
        self.IFE = entity_args['IFE']
        self.rho_e = entity_args['rho_e']
        self.sigma_e = entity_args['sigma_e']
        self.action_dim = entity_args['action_shape']

        self.real_asset_pool, self.real_e_pool, self.real_weight_pool = self.get_real_data()

        self.at = None
        self.e = None

        self.action_space = Box(
            low=-1, high=1, shape=(self.n_households, n_government, self.action_dim), dtype=np.float32
        )

        self.reset()

    def get_real_data(self):
        try:
            path = 'agents/data/advanced_scfp2022.csv'
            df = pd.read_csv(path)
            valid = ((df['ASSET'] >= 0) & (df['EDUC'] > 0) & (df['WGT'] > 0))
            asset = torch.tensor(df.loc[valid, 'ASSET'].values, device=self.device).float()
            educ = torch.tensor(df.loc[valid, 'EDUC'].values, device=self.device).float()
            weight = torch.tensor(df.loc[valid, 'WGT'].values, device=self.device).float()
            return asset, educ, weight
        except FileNotFoundError:
            print("Warning: Data file not found. Generating random data for testing.")
            asset = torch.rand(10000, device=self.device) * 100000
            educ = torch.rand(10000, device=self.device) * 2 + 1
            weight = torch.ones(10000, device=self.device)
            return asset, educ, weight

    def sample_real_data(self):
        """
        每次调用都会基于当前的随机种子，从数据池中重新采样。
        """
        required_size = self.n_governments * self.n_households

        indices = torch.multinomial(
            self.real_weight_pool, required_size, replacement=True
        )

        batch_asset = self.real_asset_pool[indices]
        batch_e = self.real_e_pool[indices]

        return batch_asset.reshape(self.n_governments, self.n_households), \
            batch_e.reshape(self.n_governments, self.n_households)

    def e_initial(self):
        """
        使用采样到的教育水平初始化生产能力 e。
        """
        self.e = self.e_init.unsqueeze(-1).clone()

    def generate_e_ability(self):
        """
        按 AR(1) 过程生成下一时刻的生产能力。
        """
        safe_e = torch.clamp(self.e[:, :, 0], min=1e-6)
        noise = torch.randn_like(safe_e) * self.sigma_e
        next_e = torch.exp(self.rho_e * torch.log(safe_e) + noise)
        self.e = torch.clamp(next_e, 0.1, 20.0).unsqueeze(-1)

    def reset(self, **custom_cfg):
        """
        每次 Reset 时重新采样数据并重新初始化过程。
        只要外部 torch.manual_seed 设置了，这里的 sample_real_data 结果就是固定的。
        """
        self.at_init, self.e_init = self.sample_real_data()

        self.e_initial()

        self.at = self.at_init.clone()
        self.at_next = self.at.clone()

    def close(self):
        pass
