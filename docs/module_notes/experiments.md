# experiments

Owns seeded deal generation, benchmark suite management, paired A/B comparison, repeated suite variance checks, regression packs, and optional external oracle adapters.

This module is first-class because parameter tuning and variance measurement are central to the project.

It now also owns the first hidden-information baseline: PIMC over uniform
determinizations. This baseline:

- samples full worlds uniformly from `BeliefState`
- evaluates candidate root actions against sampled worlds
- can reuse one shared world batch across all root actions
- delegates continuation value to the deterministic open-card solver
- reports per-action mean, variance, standard error, and deterministic node use

The benchmark runner generates reproducible seed suites and supports
single-config runs, paired A/B comparisons on identical seeds, and repeated
suite comparisons.

Preset constructors provide ready-to-run config bundles:

- `pimc_baseline()`
- `belief_uct_default()`
- `belief_uct_late_exact()`
- `fast_benchmark()`
- `balanced_benchmark()`
- `quality_benchmark()`

Each preset expands into a solver config, autoplay config, and experiment
defaults, and can be converted into root-only or full-autoplay benchmark configs.

Preset intent:

- `pimc_baseline`: uniform-determinization baseline; useful as a simple
  reference point, not the final planner.
- `belief_uct_default`: belief UCT without late-exact, calibrated down from the
  original heavier default to keep root visits practical while preserving exact
  reveal handling.
- `belief_uct_late_exact`: same planner with the hidden <= 8 exact-assignment
  regime enabled for top actions.
- `fast_benchmark`: smoke tests and quick local regressions.
- `balanced_benchmark`: the default local tuning preset, between fast and
  quality in simulation count, leaf samples, and late-exact assignment budget.
- `quality_benchmark`: slower comparison runs where stronger root estimates are
  worth the extra time.

There are now two benchmark layers:

- **Root-only benchmarks:** evaluate one recommendation from a generated belief
  root. These are fast and useful for tuning root evaluators such as PIMC.
- **Full autoplay benchmarks:** start from the true seeded full deal, derive the
  public belief state, repeatedly ask a selected backend for a move, apply that
  move to the true full state, and update the belief with the actual reveal
  observation.

Full autoplay supports `Pimc`, `BeliefUct`, and `BeliefUctLateExact` backends.
After every real reveal, the observed card is removed from `UnseenCardSet`, so
the belief remains the exact uniform posterior over the remaining face-down
tableau cards.

Full-game records report win/loss, termination reason, moves played, total and
per-move planner time, deterministic node use, root visits/samples, and
late-exact trigger counts. Paired A/B autoplay comparisons use the exact same
seed suite for both configurations, and repeated comparisons generate
deterministic independent suites from a base seed.

Exports are machine-friendly and deterministic:

- JSON summary exports use stable serde field order from summary report structs.
- CSV summary exports produce one-row benchmark or comparison summaries.
- Per-game CSV exports produce one row per autoplay game with seed, outcome,
  planner timing, deterministic nodes, root visits/samples, and late-exact
  trigger counts.

Paired comparison reports include both config names/backends, wins for A and B,
candidate-minus-baseline paired win difference, same-outcome counts, and a simple
CI-like interval from paired per-seed deltas.

Preset comparison reports evaluate many presets on the same suite and rank them
by win rate, time per game, or win rate per planner-second. The efficiency metric
is intentionally simple: `win_rate / average_planner_seconds_per_game`.

The ML export pipeline reuses the same seeded autoplay machinery for V-Net
dataset generation. `dataset export-vnet` records full deterministic states,
labels, chosen-action metadata, and preset/suite provenance as JSONL, keeping
benchmark reproducibility and training-data provenance aligned.

## Running benchmarks from the CLI

The `solitaire-cli benchmark` commands are thin wrappers around this library
module. They use deterministic seed suites and the same preset names exposed by
`experiment_preset_by_name`.

Available presets:

- `pimc_baseline`
- `belief_uct_default`
- `belief_uct_late_exact`
- `fast_benchmark`
- `balanced_benchmark`
- `quality_benchmark`

Example commands:

```powershell
cargo run -p solitaire-cli -- benchmark autoplay --preset fast_benchmark --games 25 --seed 100 --json reports/autoplay.json --csv reports/autoplay.csv --game-csv reports/autoplay-games.csv
cargo run -p solitaire-cli -- benchmark compare --baseline pimc_baseline --candidate belief_uct_late_exact --games 25 --seed 100 --json reports/compare.json --csv reports/compare.csv
cargo run -p solitaire-cli -- benchmark repeated-compare --baseline belief_uct_default --candidate belief_uct_late_exact --games 25 --repetitions 5 --seed 100 --json reports/repeated.json --csv reports/repeated.csv
cargo run -p solitaire-cli -- benchmark compare-presets --presets fast_benchmark,balanced_benchmark,quality_benchmark --games 25 --seed 100 --rank-by efficiency --json reports/presets.json --csv reports/presets.csv
cargo run -p solitaire-cli -- dataset export-vnet --preset fast_benchmark --games 25 --seed 100 --max-steps 20 --out data/vnet.jsonl
```

This module must remain honest about PIMC's limits. It is not sparse UCT, POMCP,
late-exact assignment search, or a weighted posterior model.
