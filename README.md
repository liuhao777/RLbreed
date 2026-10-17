# Structured PPO for mating allocation

Core policy code for mating and offspring allocation in rainbow trout breeding simulations.

## Modules

- `features.py`: six pair features and twelve population statistics.
- `policy.py`: independent actor and critic encoders with masked mean/max pooling.
- `decoder.py`: legal-pair ranking, softmax allocation and integer offspring counts.
- `training.py`: balanced initialization and a fresh, fixed-budget PPO run.

The actor produces six Gaussian controls. Execution clips them to [0, 1]; the
pair-score weights are `2*u - 1`, with 0.5 subtracted from the first two weights.
The initial mean is `[0.5, 0.5, 0.5, 0.25, 0.5, 0.5]`, with standard deviation 0.15.
The actor and critic each encode pairs with a 6–32–32 network, pool to 64 values,
and append 12 global statistics. Their output heads use 64 and 32 hidden units.

## Environment interface

The breeding simulator and biological assets are not included. Supply a Gymnasium
environment through `module:factory`, where the factory takes no arguments.
Its unwrapped `cfg` must provide `founders=75` and `generations=10`.
Its observation space is a float32 vector of length `75*75*7 + 12`; its action
space is `Box(0, 1, shape=(6,))`.

`features()` reads the parent genome `(n, markers, 2)`, `predicted_bs` `(n,)`,
`role_code` `(n,)`, genomic relationship `(n,n)` and `generation` `(1,)`.
`predicted_bs` is the historical key for the additive genomic score s; lower is
preferred and R = −s. Role 1 is sire and role 0 is dam. All arrays must share the
same parent order. `generation` must use the simulator's original normalized coding.
The scaling inputs to `features()` come from the fixed founder reference.

The environment converts controls with `pair_action()`, then calls
`decode_cohort_pair_scores()` with 75 cross slots and 2,599 offspring. It generates
offspring, scores them with the frozen GBLUP scorer and retains the 75 lowest-score
offspring for generations 1–9. Generation 10 ends before selection.
Intermediate reward is zero; terminal reward is
`(founder_mean_score - terminal_mean_score) / founder_score_sd`.
Hidden QTL effects and simulated mortality risk must not enter observations or reward.

The decoder retains the original ceil-and-truncate rule, including stable index
tie-breaking. It is not largest-remainder rounding and has no parent contribution cap.
Pass roles as `SIRE`/`DAM` strings and genomic relationship divided by two to the
decoder. Native PPO likelihoods use the original Gaussian samples; do not replace
the training actions with their clipped execution values.

## Usage

Use Python 3.11. Install the requirements in a separate environment; choose a PyTorch
build suitable for the local hardware. The study used PyTorch 2.3.1, SB3 2.7.0 and
Gymnasium 1.2.2. CPU is the default.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m trout_ppo.training --help
python -m trout_ppo.training --env-factory your_simulator:make_env --seed 166 --steps 2000 --output runs/ppo.zip
```

`your_simulator:make_env` is an interface placeholder, not a bundled simulator.
The training entry starts a new run. Historical checkpoint recovery, RNG archives
and provenance manifests are not included; this package cannot reproduce the
published checkpoint or evaluation paths on its own.

No data, weights, evaluation results, baseline implementations or manuscript files
are distributed.

## License

This project is released under the BSD 3-Clause License. See `LICENSE` for details.
