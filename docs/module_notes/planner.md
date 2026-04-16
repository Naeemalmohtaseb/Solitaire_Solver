# planner

Owns the first event-driven belief-state root planner.

This initial planner is root-focused sparse UCT over `BeliefState`. Legal moves
come from the existing visible move generator. Non-reveal actions use exact
deterministic belief transitions. Reveal actions expand the exact reveal
frontier and sample one equal-probability child during each simulation.

Leaf evaluation reuses the deterministic open-card solver on uniformly sampled
full worlds from the current belief state. No weighted posterior machinery is
introduced: hidden uncertainty remains exactly uniform over assignments of
unseen cards to face-down tableau slots.

This is closer to the final architecture than the PIMC baseline because reveal
nodes are explicit and non-reveal corridors remain in belief space. It is still
an initial version: late-game exact assignment mode, belief-state TT, subtree
reuse across real turns, and neural guidance remain deferred.

Root search now has three conservative control layers:

- **Root screening:** after an initial simulation phase, actions can be narrowed
  out only when they are outside the configured active-action cap and trail the
  current best mean by at least `drop_margin`.
- **Confidence stopping:** after `min_simulations_before_stop`, the planner can
  stop early when the best action's lower CI-like bound beats every active
  alternative's upper bound by `separation_margin`.
- **Reveal-aware refinement:** root reveal actions cache the exact reveal
  frontier and cycle through reveal children for fair coverage. A small optional
  second-reveal refinement pass can spend extra simulations on top close
  reveal-causing contenders.
- **Late-exact handoff:** when hidden tableau cards are at or below the
  configured threshold, default `<= 8`, the planner can hand the top root actions
  to `late_exact` for exact hidden-assignment aggregation. This replaces noisy
  sampled estimates only for the top actions, not for every legal move.

Still deferred: large belief-state transposition tables, tree reuse across real
turns, neural priors/evaluators, parallel root workers, and stronger
branch-and-bound pruning inside late-exact.
