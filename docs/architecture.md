# Solitaire Solver Architecture

This repository is a Rust-first backend for solving Draw-3 Klondike Solitaire under hidden tableau information.

The domain assumptions are fixed:

- Stock and waste order are fully known from the start.
- Hidden uncertainty exists only in face-down tableau cards.
- Legal actions depend only on the visible state.
- Hidden cards branch only when a move reveals a tableau card.
- The posterior over hidden tableau assignments is exactly uniform over all assignments consistent with the visible state.

The engine is organized around a strong perfect-information solver first, with belief-state planning layered above it.

## Runtime Layers

1. `core` and `cards` define state, card identity, tableau columns, and invariants.
2. `stock` models exact Draw-3 stock/waste cycling.
3. `moves` will own legal atomic moves, macro moves, and apply/undo.
4. `closure` will compress deterministic corridors and stop at reveal or meaningful decision frontiers.
5. `hashing_tt` will provide Zobrist hashing, canonicalization, and transposition tables.
6. `deterministic_solver` will solve or evaluate fully instantiated open-card worlds.
7. `belief` will handle reveal chance nodes and uniform hidden-tableau sampling.
8. `planner` will run event-driven sparse UCT over belief states.
9. `late_exact` will enumerate hidden assignments when few hidden cards remain.
10. `experiments` will run seeded deals, paired A/B comparisons, and regression benchmarks.
11. `session` will persist real-game state and support subtree reuse.
12. `ml` will later export data and connect small V-Net/P-Net models.

## Current Prompt Scope

Prompt 1 creates only architecture scaffolding:

- Cargo workspace
- module tree
- configuration structs
- error types
- skeletal state/result surfaces
- documentation placeholders
- minimal CLI diagnostics

It intentionally does not implement move generation, search, heuristics, or solver behavior.

