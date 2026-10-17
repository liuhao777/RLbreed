import numpy as np
from ._validation import require


def compact(obs, mean0, sd0, h0, freq0):
    scores = np.asarray(obs["predicted_bs"], float)
    genome = np.asarray(obs["genome"])
    roles = obs["role_code"]
    legal = (roles[:, None] > 0.5) & (roles[None, :] < 0.5)
    kin = np.asarray(obs["genomic_relationship"], float)[legal] / 2
    require(len(kin) > 0, "No legal parent pairs")
    h = float(np.mean(genome[:, :, 0] != genome[:, :, 1]))
    freq = genome.mean(axis=(0, 2))
    out = np.asarray(
        [
            float(obs["generation"][0]),
            (scores.mean() - mean0) / sd0,
            scores.std() / sd0,
            (np.quantile(scores, 0.1) - mean0) / sd0,
            (np.quantile(scores, 0.9) - mean0) / sd0,
            h / h0,
            kin.mean(),
            kin.std(),
            kin.min(),
            kin.max(),
            roles.mean(),
            np.mean(np.abs(freq - freq0)),
        ],
        np.float32,
    )
    require(out.shape == (12,) and np.isfinite(out).all(), "Bad compact observation")
    return out


def zscore(x):
    x = np.asarray(x, float)
    return np.clip((x - x.mean()) / max(float(x.std()), 1e-6), -3.0, 3.0)


def pair_basis(obs):
    s = np.asarray(obs["predicted_bs"], float)
    role = np.asarray(obs["role_code"])
    k = np.asarray(obs["genomic_relationship"], float) / 2
    n = len(s)
    mask = (role[:, None] > 0.5) & (role[None, :] < 0.5)
    require(mask.any() and np.isfinite(s).all() and np.isfinite(k).all(), "Invalid parents")
    z = zscore(s)
    sire = np.broadcast_to(z[:, None], (n, n))
    dam = np.broadcast_to(z[None, :], (n, n))
    sire_k = np.broadcast_to(k[:, role < 0.5].mean(1)[:, None], (n, n))
    dam_k = np.broadcast_to(k[role > 0.5, :].mean(0)[None, :], (n, n))
    fields = [sire, dam, abs(sire - dam), k, sire_k, dam_k]

    out = np.zeros((n, n, 6), np.float32)
    for j, x in enumerate(fields):
        out[:, :, j][mask] = zscore(x[mask])
    return out, mask


def features(obs, mean0, sd0, h0, freq0):
    pair, mask = pair_basis(obs)
    glob = compact(obs, mean0, sd0, h0, freq0)
    return np.concatenate([pair.ravel(), mask.ravel().astype(np.float32), glob]).astype(
        np.float32
    )


def pair_action(obs, controls):
    u = np.asarray(controls, float)
    require(
        u.shape == (6,) and np.isfinite(u).all() and ((u >= 0) & (u <= 1)).all(),
        "Invalid six-control action",
    )
    f, mask = pair_basis(obs)

    w = 2 * u - 1
    w = w.copy()
    w[:2] -= 0.5
    a = np.einsum("ijk,k->ij", f, w).astype(np.float32)
    a[~mask] = -1e5
    return a
