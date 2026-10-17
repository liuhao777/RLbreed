import argparse
import importlib
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from ._validation import require
from .policy import new_model

TRAINING = dict(
    n_steps=20,
    batch_size=10,
    n_epochs=4,
    learning_rate=1e-4,
    gamma=1.0,
    gae_lambda=0.95,
    target_kl=0.02,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)


class EpisodeStream(gym.Wrapper):
    def __init__(self, env, seed):
        super().__init__(env)
        self.training_seed = int(seed)
        self.next_episode = 0

    def reset(self, *, seed=None, options=None):
        require(0 <= self.next_episode < 10000, "Training stream exhausted")
        episode_seed = 2000000000 + 10000 * self.training_seed + self.next_episode
        self.next_episode += 1
        return self.env.reset(seed=episode_seed, options=options)


def train(make_env, output, seed, steps=2000, device="cpu"):
    require(
        isinstance(steps, int) and steps > 0 and steps % 20 == 0,
        "Training budget must be a positive multiple of 20",
    )
    output = Path(output)
    require(output.suffix == ".zip", "Output must use the .zip extension")
    require(not output.exists(), "Output already exists")
    require(device != "cuda" or torch.cuda.is_available(), "CUDA is not available")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    env = make_env()
    try:
        cfg = env.unwrapped.cfg
        require(cfg["generations"] == 10, "Expected a ten-generation environment")
        require(cfg["founders"] == 75, "Expected 75 breeding parents")
        require(env.action_space.shape == (6,), "Expected six bounded controls")
        require(
            np.all(env.action_space.low == 0) and np.all(env.action_space.high == 1),
            "Expected controls bounded to [0, 1]",
        )
        model = new_model(EpisodeStream(env, seed), seed, device, dict(TRAINING))
        model.learn(total_timesteps=steps)
        require(model.num_timesteps == steps, "Training budget mismatch")
        output.parent.mkdir(parents=True, exist_ok=True)
        model.save(output)
        return model
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-factory", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    module, name = args.env_factory.split(":", 1)
    factory = getattr(importlib.import_module(module), name)
    train(factory, args.output, args.seed, args.steps, args.device)


if __name__ == "__main__":
    main()
