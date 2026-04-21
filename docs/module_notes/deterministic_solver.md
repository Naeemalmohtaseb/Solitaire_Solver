# deterministic_solver

Owns the perfect-information open-card solver used by later belief-state
planning.

The production module keeps the public result/config/TT/search surfaces together
because those pieces are tightly coupled in the current DFS implementation.
Regression tests live in `deterministic_solver/tests.rs` so the solver module is
easier to scan without changing its public API.

The first implementation is a deterministic DFS-based searcher with:

- exact/proof-oriented solve attempts
- bounded solve mode
- fast heuristic evaluation
- explicit node/depth/time budgets
- structured diagnostics
- deterministic move ordering
- closure integration at every node
- deterministic-state transposition-table reuse
- optional V-Net approximate leaf evaluation

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

Move ordering and heuristic cutoff evaluation use a small explainable strategic
score. This is not pruning. It only orders children and shapes approximate leaf
values. The current principles are:

- reveal-causing moves first because they reduce hidden information
- clearly safe Ace/Two foundation moves very early
- other foundation progress before noisy tableau rearrangement
- stock/waste access improvements before low-information tableau churn
- empty columns are useful when they create real King space or pair with reveal
  progress
- visible-run moves that only reshuffle tableau cards without reveal,
  foundation, stock, or empty-column progress receive a conservative churn
  penalty
- foundation retreats stay legal, but are ordered late unless search proves
  they matter

The handcrafted fast evaluator mirrors those principles by valuing foundation
progress, hidden-card reduction, reveal potential, safe foundation availability,
productive mobility, stock access, and useful empty columns, while discounting
fake mobility from reversible tableau shuffles.

`LeafEvaluationMode::VNet` can replace the handcrafted cutoff evaluator with a
loaded V-Net artifact for fast/bounded leaf values. This is deliberately
approximate: exact/proof-oriented outcomes still require proven search results,
and V-Net values are never allowed to masquerade as proof. If no evaluator is
loaded, the solver falls back to the existing heuristic and records fallback
diagnostics.

No belief-state sampling, UCT/MCTS, P-Net policy logic, online training, or
late-game exact assignment enumeration belongs in this module.
