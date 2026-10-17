import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from ._validation import require

BALANCED = [0.5, 0.5, 0.5, 0.25, 0.5, 0.5]


class PairSetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, n):
        super().__init__(observation_space, features_dim=76)
        self.n = int(n)
        self.pairs = self.n * self.n
        require(
            observation_space.shape == (self.pairs * 7 + 12,),
            "Wrong structured observation shape",
        )
        self.encoder = nn.Sequential(nn.Linear(6, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())

    def forward(self, x):
        f = x[:, : self.pairs * 6].reshape(-1, self.pairs, 6)
        mask = x[:, self.pairs * 6 : self.pairs * 7].reshape(-1, self.pairs, 1) > 0.5
        h = self.encoder(f)
        count = mask.sum(1).clamp_min(1)
        avg = torch.where(mask, h, 0.0).sum(1) / count
        maximum = torch.where(mask, h, torch.full_like(h, -1e6)).max(1).values
        return torch.cat([avg, maximum, x[:, -12:]], dim=1)


def new_model(env, seed, device, config):
    m = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        device=device,
        verbose=0,
        policy_kwargs=dict(
            features_extractor_class=PairSetExtractor,
            features_extractor_kwargs=dict(n=env.unwrapped.cfg["founders"]),
            net_arch=dict(pi=[64, 32], vf=[64, 32]),
            share_features_extractor=False,
            log_std_init=float(np.log(0.15)),
        ),
        **config
    )
    with torch.no_grad():
        m.policy.action_net.bias.fill_(0.5)
    with torch.no_grad():
        m.policy.action_net.weight.zero_()
        m.policy.action_net.bias.copy_(
            torch.as_tensor(BALANCED, dtype=m.policy.action_net.bias.dtype, device=m.device)
        )
    return m
