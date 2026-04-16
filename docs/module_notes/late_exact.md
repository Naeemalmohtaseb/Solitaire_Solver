# late_exact

Owns the late-game exact hidden-assignment regime.

This is a small-hidden-count regime switch. When the belief state has at most
the configured number of hidden tableau cards, default `<= 8`, this subsystem
enumerates assignments of `BeliefState.unseen_cards` to
`VisibleState.hidden_slots()` in deterministic slot/card order.

For each enumerated assignment, it builds a `FullState`, applies a candidate
root move through the existing full-state move engine, and delegates continuation
value to the deterministic open-card solver. Aggregation is a straight uniform
mean over assignments. There is no weighted posterior machinery.

The v1 implementation supports recursive prefix enumeration and an optional
assignment budget. Conservative pruning hooks are represented in diagnostics,
but no speculative pruning is applied yet.

Action evaluation streams assignment traversal instead of materializing every
assignment up front. The public enumeration helper remains available for tests
and diagnostics, while planner-triggered late-exact evaluation keeps memory
bounded and reports assignment traversal time separately from deterministic
solver work.

Planner integration is intentionally narrow: late-exact is applied only for
eligible small-hidden states and only to the configured number of top root
actions, default two. It replaces noisy sampled estimates for those top actions
without adding a belief-state TT or changing the planner architecture.
