# Roadmap

## Prompt 1: Workspace and Scaffolding

Create a compile-ready Rust workspace with clear module boundaries, configuration surfaces, error types, skeletal domain types, docs, and a minimal CLI.

## Prompt 2: Core Domain Model

Implement card/state validation, tableau column invariants, visible/full/belief state construction, and pretty-printing.

## Prompt 3: Stock/Waste Model

Implement exact Draw-3 stock/waste cycling, accessibility, recycle behavior, and canonical stock state representation.

## Prompt 4: Move Logic

Implement legal atomic moves, macro moves, apply/undo, auto-flip side effects, and exhaustive legality tests.

## Prompt 5: Closure and Dominance Framework

Implement deterministic corridor compression, inverse suppression, safe symmetry hooks, move-ordering scores, and debug traces.

## Prompt 6: Hashing and Transposition Tables

Implement Zobrist hashing, canonicalization, deterministic TT, belief-node cache, and collision-audit hooks.

## Prompt 7: Deterministic Open-Card Solver

Implement the DFS/best-first hybrid solver, bounded search, heuristic fallback evaluator, and diagnostics.

## Prompt 8: Belief Layer

Implement uniform hidden-tableau sampling, deterministic belief transitions, exact first-reveal expansion, and second-reveal helper logic.

## Prompt 9: Event-Driven Planner

Implement root action screening, sparse UCT, reveal chance nodes, branch-priority budgeting, early stopping, and root-worker orchestration.

## Prompt 10: Late Exact Assignment

Implement hidden-card assignment enumeration for late-game positions, expected-value aggregation, branch-and-bound, and TT reuse.

## Prompt 11: Experiments

Implement seeded deal suites, paired A/B runner, repeated suite comparison, regression packs, reports, and optional oracle adapters.

## Prompt 12: Session and ML Hooks

Implement save/load, move logs, subtree reuse metadata, and dataset export surfaces for later V-Net/P-Net training.

