# ml

Owns machine-learning data boundaries without adding neural inference or training code.

The current implementation exports V-Net supervised examples from fully known
`FullState` positions. Each example contains a deterministic structured encoding,
a fixed per-card numeric feature vector, an explicit scalar label, and provenance
metadata linking it back to a preset, deal seed, and autoplay step.

## V-Net Labels

Supported label modes are explicit and must not be mixed silently:

- `TerminalOutcome`: labels every exported state by the final win/loss of the configured autoplay run.
- `DeterministicSolverValue`: labels a full state with the deterministic open-card solver's bounded value.
- `PlannerBackedApproximateValue`: labels decision states with the planner root value when available.

## Export Format

The v1 writer emits JSONL:

1. one metadata record
2. one example record per exported full state

The CLI entry point is:

```text
solitaire-cli dataset export-vnet --preset fast_benchmark --games 10 --seed 1 --max-steps 20 --out data/vnet.jsonl
```

## Deferred

PyTorch datasets, V-Net model definitions, inference adapters, P-Net exports, and
training scripts remain deferred. This module is only the deterministic data
export pipeline.
