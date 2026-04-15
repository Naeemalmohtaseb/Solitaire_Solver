# moves

Owns atomic moves, planner macro moves, legal move generation, apply/undo, auto-flip side effects, and semantic move tags.

Move application is visible-state based and reversible. Automatic tableau reveals are transition side effects, represented in move outcomes and undo records, not separate player actions.

Future search layers should consume macro moves and semantic summaries, but atomic move correctness remains the foundation.
