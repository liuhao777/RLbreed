from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CrossPlanEntry:
    sire_index: int
    dam_index: int
    n_offspring: int
    pair_score: float
    expected_progeny_inbreeding: float


def decode_cohort_pair_scores(
    action: np.ndarray,
    roles: np.ndarray,
    n_crosses: int,
    n_candidate_offspring: int,
    expected_progeny_inbreeding: np.ndarray | None = None,
    max_expected_progeny_inbreeding: float | None = None,
    score_low: float = -1e5,
    score_high: float = 1e5,
) -> list[CrossPlanEntry]:

    action = np.asarray(action, dtype=np.float64)
    roles = np.asarray(roles)
    n = int(roles.size)
    if action.shape != (n, n):
        raise ValueError(f"action shape must be {(n, n)}")
    if n_crosses <= 0 or n_candidate_offspring <= 0:
        raise ValueError("cohort counts must be positive")
    indices = np.arange(n)
    legal = (
        (roles[:, None] == "SIRE")
        & (roles[None, :] == "DAM")
        & (indices[:, None] != indices[None, :])
    )
    finite = np.isfinite(action) & (action >= score_low) & (action <= score_high)
    eligible = legal & finite
    pair_inbreeding = None
    if expected_progeny_inbreeding is not None:
        pair_inbreeding = np.asarray(expected_progeny_inbreeding, dtype=np.float64)
        if pair_inbreeding.shape != (n, n) or not np.isfinite(pair_inbreeding).all():
            raise ValueError("expected progeny inbreeding matrix is invalid")
    if max_expected_progeny_inbreeding is not None:
        if pair_inbreeding is None:
            raise ValueError("coancestry constraint requires a pair inbreeding matrix")
        eligible &= pair_inbreeding <= float(max_expected_progeny_inbreeding)
    flat = np.flatnonzero(eligible.ravel())
    if flat.size < n_crosses:
        if max_expected_progeny_inbreeding is not None:
            raise ValueError(
                "coancestry constraint is infeasible: "
                f"only {flat.size} eligible pairs; need {n_crosses}"
            )
        raise ValueError(f"only {flat.size} legal finite pairs; need {n_crosses}")
    values = action.ravel()[flat]
    order = np.lexsort((flat, -values))[:n_crosses]
    selected = flat[order]
    selected_values = action.ravel()[selected]
    shifted = selected_values - np.max(selected_values)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    counts = np.ceil(weights * n_candidate_offspring).astype(np.int64)
    expanded = np.repeat(np.arange(n_crosses, dtype=np.int64), counts)
    if expanded.size == 0:
        raise RuntimeError("cohort allocation produced no realized cross")
    if expanded.size >= n_candidate_offspring:
        expanded = expanded[:n_candidate_offspring]
    else:
        expanded = np.pad(
            expanded,
            (0, n_candidate_offspring - expanded.size),
            mode="constant",
            constant_values=int(expanded[-1]),
        )
    allocated = np.bincount(expanded, minlength=n_crosses)
    plan: list[CrossPlanEntry] = []
    for rank, count in enumerate(allocated):
        if count <= 0:
            continue
        sire, dam = np.unravel_index(int(selected[rank]), (n, n))
        expected_f = (
            float("nan") if pair_inbreeding is None else float(pair_inbreeding[sire, dam])
        )
        plan.append(
            CrossPlanEntry(
                int(sire),
                int(dam),
                int(count),
                float(selected_values[rank]),
                expected_f,
            )
        )
    if sum(entry.n_offspring for entry in plan) != n_candidate_offspring:
        raise RuntimeError("cohort allocation did not close after reference padding/truncation")
    return plan


def stable_truncation_selection(scores: np.ndarray, k: int) -> np.ndarray:

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite vector")
    if not 0 < k <= values.size:
        raise ValueError("selection k is outside candidate cohort")
    return np.lexsort((np.arange(values.size), values))[:k].astype(np.int64)
