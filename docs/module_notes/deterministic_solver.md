# deterministic_solver

Owns the perfect-information open-card solver used by later belief-state
planning.

The first implementation is a deterministic DFS-based searcher with:

- exact/proof-oriented solve attempts
- bounded solve mode
- fast heuristic evaluation
- explicit node/depth/time budgets
- structured diagnostics
- deterministic move ordering
- closure integration at every node
- deterministic-state transposition-table reuse

The solver operates on `FullState`. When a move reveals a hidden tableau card,
the concrete card is read from `HiddenAssignments`, the visible move engine is
called with that reveal card, and the assignment is removed. Undo restores both
the visible transition and the hidden assignment.

The deterministic TT is scoped strictly to fully known states. Its structural
hash key includes foundation tops, tableau hidden counts and face-up cards,
exact stock/waste state, and concrete hidden assignments. Entries distinguish
proven exact win/loss values from approximate bounded-search values; exact mode
only reuses exact entries as proof. Approximate entries can still provide value
reuse in bounded/fast modes and best-move hints for deterministic move ordering.

The current table is direct-mapped with a simple bounded capacity and
priority-based replacement. Zobrist hashing and richer bound handling can
replace the structural hash/interface later without changing planner-facing
solver APIs.

No belief-state sampling, UCT/MCTS, neural inference, or late-game exact
assignment enumeration belongs in this module.
