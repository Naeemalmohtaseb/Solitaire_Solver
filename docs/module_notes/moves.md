# moves

Owns atomic moves, planner macro moves, legal move generation, apply/undo, auto-flip side effects, and semantic move tags.

The public move engine remains in one production module for now because move
generation, apply/undo, stock changes, and full-state reveal bridging share the
same invariants. Tests live in `moves/tests.rs` to keep the hot-path code easier
to navigate.

Move application is visible-state based and reversible. Automatic tableau reveals are transition side effects, represented in move outcomes and undo records, not separate player actions.

Future search layers should consume macro moves and semantic summaries, but atomic move correctness remains the foundation.
