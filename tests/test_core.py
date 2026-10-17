import copy
import tempfile
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from trout_ppo.decoder import decode_cohort_pair_scores, stable_truncation_selection
from trout_ppo.features import features, pair_action, pair_basis
from trout_ppo.policy import BALANCED, PairSetExtractor, new_model
from trout_ppo.training import EpisodeStream, TRAINING, train


def observation(n=75):
    rng = np.random.default_rng(9)
    genome = rng.integers(0, 2, (n, 32, 2), dtype=np.int8)
    matrix = rng.normal(size=(n, n))
    return {
        "predicted_bs": rng.normal(size=n).astype(np.float32),
        "role_code": (np.arange(n) % 2).astype(np.int8),
        "genomic_relationship": ((matrix + matrix.T) / 2).astype(np.float32),
        "genome": genome,
        "generation": np.array([-1.0], dtype=np.float32),
    }


def encoded(obs):
    return features(obs, 0.0, 1.0, 0.5, obs["genome"].mean(axis=(0, 2)))


class InputEnv(gym.Env):
    def __init__(self):
        self.cfg = {"founders": 75, "generations": 10}
        self.value = encoded(observation())
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=self.value.shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(6,), dtype=np.float32)
        self.received_seed = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.received_seed = seed
        return self.value.copy(), {}

    def step(self, action):
        raise AssertionError("Tests must not run breeding transitions")


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_feature_shape_and_mask(self):
        obs = observation()
        pair, mask = pair_basis(obs)
        self.assertEqual(pair.shape, (75, 75, 6))
        self.assertEqual(encoded(obs).shape, (39387,))
        self.assertTrue(np.isfinite(encoded(obs)).all())
        self.assertTrue((pair[~mask] == 0).all())
        self.assertLessEqual(np.max(np.abs(pair)), 3.0)

    def test_balanced_scores(self):
        obs = observation()
        pair, mask = pair_basis(obs)
        actual = pair_action(obs, BALANCED)
        expected = -0.5 * (pair[:, :, 0] + pair[:, :, 1] + pair[:, :, 3])
        np.testing.assert_allclose(actual[mask], expected[mask], atol=1e-6)
        self.assertTrue((actual[~mask] == -1e5).all())

    def test_invalid_controls(self):
        obs = observation()
        for controls in [[0] * 5, [-1] * 6, [2] * 6, [np.nan] * 6]:
            with self.assertRaises(RuntimeError):
                pair_action(obs, controls)

    def test_parent_permutation(self):
        obs = observation()
        order = np.random.default_rng(3).permutation(75)
        other = copy.deepcopy(obs)
        for key in ["predicted_bs", "role_code", "genome"]:
            other[key] = other[key][order]
        other["genomic_relationship"] = other["genomic_relationship"][order][:, order]
        controls = [0.1, 0.7, 0.5, 0.2, 0.8, 0.3]
        np.testing.assert_allclose(
            pair_action(other, controls), pair_action(obs, controls)[order][:, order], atol=1e-6
        )
        net = PairSetExtractor(InputEnv().observation_space, 75)
        with torch.no_grad():
            a = net(torch.as_tensor(encoded(obs)[None]))
            b = net(torch.as_tensor(encoded(other)[None]))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_pool_ignores_illegal_pairs(self):
        env = InputEnv()
        x = env.value.copy()
        count = 75 * 75
        mask = x[count * 6 : count * 7] > 0.5
        changed = x.copy()
        changed[: count * 6].reshape(count, 6)[~mask] = 100.0
        net = PairSetExtractor(env.observation_space, 75)
        with torch.no_grad():
            out = net(torch.as_tensor(np.stack([x, changed])))
        torch.testing.assert_close(out[0], out[1])

    def test_native_policy_initialization_and_serialization(self):
        env = InputEnv()
        model = new_model(env, 166, "cpu", dict(TRAINING))
        self.assertIsNot(model.policy.pi_features_extractor, model.policy.vf_features_extractor)
        for value in [env.value, env.value * 0.7]:
            action, _ = model.predict(value, deterministic=True)
            np.testing.assert_array_equal(action, BALANCED)
        np.testing.assert_allclose(model.policy.log_std.detach().exp().numpy(), 0.15, atol=1e-8)
        self.assertEqual(model.num_timesteps, 0)
        self.assertFalse(model.policy.optimizer.state)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "policy.zip"
            model.save(target)
            loaded = PPO.load(target, device="cpu")
            action, _ = loaded.predict(env.value, deterministic=True)
            np.testing.assert_array_equal(action, BALANCED)
        env.close()

    def test_decoder_count_and_roles(self):
        obs = observation()
        roles = np.where(obs["role_code"] > 0.5, "SIRE", "DAM")
        plan = decode_cohort_pair_scores(
            pair_action(obs, BALANCED),
            roles,
            75,
            2599,
            obs["genomic_relationship"] / 2,
        )
        self.assertEqual(sum(row.n_offspring for row in plan), 2599)
        self.assertLessEqual(len(plan), 75)
        for row in plan:
            self.assertEqual(roles[row.sire_index], "SIRE")
            self.assertEqual(roles[row.dam_index], "DAM")
            self.assertGreater(row.n_offspring, 0)

    def test_decoder_stable_tie_and_selection(self):
        roles = np.array(["SIRE", "SIRE", "DAM", "DAM"])
        plan = decode_cohort_pair_scores(np.zeros((4, 4)), roles, 3, 10)
        self.assertEqual(
            [(r.sire_index, r.dam_index, r.n_offspring) for r in plan],
            [(0, 2, 4), (0, 3, 4), (1, 2, 2)],
        )
        np.testing.assert_array_equal(
            stable_truncation_selection([0.2, -1, -1, 0.1], 2), [1, 2]
        )
        with self.assertRaises(ValueError):
            decode_cohort_pair_scores(np.zeros((4, 4)), np.array(["DAM"] * 4), 3, 10)

    def test_episode_stream(self):
        env = InputEnv()
        stream = EpisodeStream(env, 166)
        for episode in range(3):
            stream.reset()
            self.assertEqual(env.received_seed, 2001660000 + episode)
        stream.close()

    def test_training_rejects_invalid_budget_and_output(self):
        for steps in [0, 1, 1999, -20]:
            with self.assertRaises(RuntimeError):
                train(InputEnv, "unused.zip", 166, steps=steps)
        with self.assertRaises(RuntimeError):
            train(InputEnv, "unused", 166)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "existing.zip"
            target.touch()
            with self.assertRaises(RuntimeError):
                train(InputEnv, target, 166)


if __name__ == "__main__":
    unittest.main()
