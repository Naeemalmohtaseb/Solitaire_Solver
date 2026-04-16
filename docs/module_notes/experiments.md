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
- `fast_vnet_benchmark()`
- `balanced_benchmark()`
- `balanced_vnet_benchmark()`
- `quality_benchmark()`
- `quality_vnet_benchmark()`

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
- `fast_vnet_benchmark`: same role as `fast_benchmark`, but approximate
  deterministic leaves use the V-Net path with heuristic fallback.
- `balanced_benchmark`: the default local tuning preset, between fast and
  quality in simulation count, leaf samples, and late-exact assignment budget.
- `balanced_vnet_benchmark`: a V-Net-assisted middle ground. It spends slightly
  more root simulations than the non-neural balanced preset, but uses fewer leaf
  worlds because the learned leaf is intended to be cheaper than bounded search.
- `quality_benchmark`: slower comparison runs where stronger root estimates are
  worth the extra time.
- `quality_vnet_benchmark`: a heavier V-Net-assisted preset for longer
  comparisons. It keeps exact proof semantics unchanged; V-Net is only an
  approximate cutoff evaluator.

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

Planner recommendations also expose microsecond timing counters for belief
transitions, reveal expansion, leaf evaluation, deterministic leaf solving, and
late-exact assignment traversal. Hot-path timing is opt-in via
`enable_perf_timing`; these are benchmark diagnostics only and do not affect
move selection.

Belief UCT autoplay can also enable root-parallel mode through
`BeliefPlannerConfig`. Root-parallel mode runs independent workers from the same
public belief root, with deterministic seed offsets and worker simulation
budgets, then aggregates only root action statistics. Benchmark reports include
root-parallel step counts, average workers per game, and aggregate worker
simulation counts. This is a latency/quality knob, not a new planner algorithm:
non-reveal transitions, reveal-frontier handling, and leaf evaluation semantics
are unchanged.

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

V-Net-aware calibration helpers compare heuristic and V-Net leaf modes on the
same paired autoplay suite. Reports include the configured leaf evaluation mode,
model path or artifact id, inference count, fallback count, inference time,
deterministic node count, root visits/samples, late-exact triggers, and paired
win-rate deltas. These metrics are for calibration and benchmarking only; exact
deterministic solve modes still require proof and never treat V-Net output as an
exact result.

The ML export pipeline reuses the same seeded autoplay machinery for V-Net
dataset generation. `dataset export-vnet` records full deterministic states,
labels, chosen-action metadata, and preset/suite provenance as JSONL, keeping
benchmark reproducibility and training-data provenance aligned.

## Regression Packs

Regression packs are durable, versioned JSON fixtures for hard or interesting
solver cases. They sit above the existing solver, planner, session, benchmark,
and oracle layers; they do not introduce new search behavior.

Supported case kinds:

- deterministic open-card cases
- hidden-information belief/root cases
- full-game autoplay start cases
- oracle comparison bundles
- persisted session replay cases

Each case carries explicit expectations, such as deterministic outcome,
deterministic best move, planner chosen move for a named preset, autoplay
termination, replay consistency, oracle mismatch count, or approximate value
with a tolerance. Approximate expectations stay explicit so they are not
confused with exact proof checks.

Cases can be tagged with lightweight labels such as `reveal-heavy`,
`stock-pivot`, `late-exact`, `oracle-mismatch`, `empty-column`, or
`foundation-trap`. Tags are just metadata for curation and filtering by external
tools.

Useful CLI commands:

```powershell
cargo run -p solitaire-cli -- regression create-from-benchmark --preset fast_benchmark --games 25 --seed 100 --out regression/hard.json --tag reveal-heavy
cargo run -p solitaire-cli -- regression create-from-session --path sessions/game.json --out regression/session.json --tag replay
cargo run -p solitaire-cli -- regression summarize --pack regression/hard.json
cargo run -p solitaire-cli -- regression run --pack regression/hard.json --preset fast_benchmark --json regression/run.json --csv regression/run.csv
```

Regression packs complement benchmarks and oracle comparison: benchmarks measure
aggregate behavior, oracle comparison checks external reference agreement, and
regression packs pin specific known cases so future code changes surface
behavior drift quickly.

## Offline Oracle Comparison

The oracle workflow validates the deterministic open-card solver against an
external reference without making that reference a runtime dependency and
without importing external solver code.

The library exports versioned JSON case packs:

- `OracleCasePack`
- `OracleCase`
- `OracleCaseProvenance`

Each case contains a stable `case_id`, a fully known `FullState`, provenance
metadata, and an optional expected/reference result. Cases can be generated from
seeded benchmark suites, explicit full states, or reconstructed autoplay traces.

Local evaluation uses the existing deterministic solver only:

- `OracleEvaluationMode::Exact`
- `OracleEvaluationMode::Bounded`
- `OracleEvaluationMode::Fast`

External references are ingested from simple JSON/JSONL interchange rows keyed
by `case_id`. The comparison layer reports missing rows, exact outcome
disagreements, unknown-vs-exact ambiguity, and best-move disagreements. This is
intended for offline validation against tools such as Solvitaire or curated
regression packs, but those tools remain out-of-process and optional.

## Running benchmarks from the CLI

The `solitaire-cli benchmark` commands are thin wrappers around this library
module. They use deterministic seed suites and the same preset names exposed by
`experiment_preset_by_name`.

Available presets:

- `pimc_baseline`
- `belief_uct_default`
- `belief_uct_late_exact`
- `fast_benchmark`
- `fast_vnet_benchmark`
- `balanced_benchmark`
- `balanced_vnet_benchmark`
- `quality_benchmark`
- `quality_vnet_benchmark`

Example commands:

```powershell
cargo run -p solitaire-cli -- benchmark autoplay --preset fast_benchmark --games 25 --seed 100 --json reports/autoplay.json --csv reports/autoplay.csv --game-csv reports/autoplay-games.csv
cargo run -p solitaire-cli -- benchmark compare --baseline pimc_baseline --candidate belief_uct_late_exact --games 25 --seed 100 --json reports/compare.json --csv reports/compare.csv
cargo run -p solitaire-cli -- benchmark repeated-compare --baseline belief_uct_default --candidate belief_uct_late_exact --games 25 --repetitions 5 --seed 100 --json reports/repeated.json --csv reports/repeated.csv
cargo run -p solitaire-cli -- benchmark compare-presets --presets fast_benchmark,balanced_benchmark,quality_benchmark --games 25 --seed 100 --rank-by efficiency --json reports/presets.json --csv reports/presets.csv
cargo run -p solitaire-cli -- dataset export-vnet --preset fast_benchmark --games 25 --seed 100 --max-steps 20 --decision-stride 2 --out data/vnet.jsonl
cargo run -p solitaire-cli -- benchmark autoplay --preset fast_benchmark --games 25 --seed 100 --leaf-eval-mode vnet --vnet-model python/runs/vnet-smoke/best_vnet_inference.json
cargo run -p solitaire-cli -- benchmark compare --baseline balanced_benchmark --candidate balanced_vnet_benchmark --games 25 --seed 100 --candidate-vnet-model python/runs/vnet-smoke/best_vnet_inference.json
cargo run -p solitaire-cli -- benchmark compare-presets --presets fast_benchmark,fast_vnet_benchmark,balanced_benchmark,balanced_vnet_benchmark --games 25 --seed 100 --rank-by efficiency --vnet-model python/runs/vnet-smoke/best_vnet_inference.json
cargo run -p solitaire-cli -- oracle export-cases --preset fast_benchmark --games 25 --seed 100 --out oracle/cases.json
cargo run -p solitaire-cli -- oracle evaluate-local --cases oracle/cases.json --out oracle/local.json --mode exact --node-budget 100000
cargo run -p solitaire-cli -- oracle compare --local oracle/local.json --reference oracle/reference.jsonl --json oracle/summary.json --csv oracle/summary.csv
cargo run -p solitaire-cli -- regression create-from-benchmark --preset fast_benchmark --games 25 --seed 100 --out regression/hard.json --tag stock-pivot
cargo run -p solitaire-cli -- regression run --pack regression/hard.json --preset fast_benchmark --json regression/run.json --csv regression/run.csv
```

V-Net benchmark flags are optional and only affect approximate deterministic
leaf evaluation. They do not change belief transitions, reveal-frontier
semantics, or exact proof handling. For `compare-presets`, a shared
`--vnet-model` fills model paths for presets that are already configured for
V-Net leaves; adding `--leaf-eval-mode vnet` intentionally forces every listed
preset into V-Net mode.

This module must remain honest about PIMC's limits. It is not sparse UCT, POMCP,
late-exact assignment search, or a weighted posterior model.
