# planner

Owns the first event-driven belief-state root planner.

Production code is split by concern:

- `mod.rs`: public planner API, core single-worker simulation loop, reuse cache
  types, and belief/leaf evaluation integration.
- `parallel.rs`: independent root-worker orchestration and root-stat
  aggregation only.
- `support.rs`: small deterministic RNG and stable hash helpers used by planner
  identity/fingerprint code.
- `tests.rs`: planner regression tests.

This initial planner is root-focused sparse UCT over `BeliefState`. Legal moves
come from the existing visible move generator. Non-reveal actions use exact
deterministic belief transitions. Reveal actions expand the exact reveal
frontier and sample one equal-probability child during each simulation.

Leaf evaluation reuses the deterministic open-card solver on uniformly sampled
full worlds from the current belief state. No weighted posterior machinery is
introduced: hidden uncertainty remains exactly uniform over assignments of
unseen cards to face-down tableau slots.

The deterministic solver can optionally use a loaded V-Net for approximate leaf
values. This does not change the planner architecture: simulations still use
exact belief transitions and exact reveal frontiers, while the sampled full-world
leaf call switches between heuristic, bounded deterministic, exact
deterministic, or V-Net-backed approximate evaluation according to config.

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
- **Lightweight continuation:** callers can use
  `recommend_move_belief_uct_with_reuse` to persist the previous root
  recommendation, root action stats, deterministic child keys, and exact
  reveal-child keys. Exact current-root matches can return the cached
  recommendation immediately. Followed moves and observed reveal children are
  recognized by belief identity and reported in diagnostics; if the next root
  was not already cached, the planner safely falls back to a fresh root search.
- **Performance counters:** recommendations expose lightweight timing fields for
  belief transitions, reveal expansion, leaf evaluation, deterministic leaf
  solver calls, and late-exact assignment traversal. Hot-path microsecond timing
  is opt-in with `enable_perf_timing`; the fields remain zero during normal
  runs. Recommendations also report V-Net inference and fallback counts when the
  deterministic leaf solver is configured for V-Net. These counters are
  diagnostics, not search-policy inputs.
- **Root-parallel workers:** when `enable_root_parallel` is set, the planner
  clones the same root `BeliefState` into independent workers. Each worker runs
  the existing single-root planner with a deterministic seed offset and a
  simulation slice. No deep tree is shared and no weighted posterior machinery is
  introduced. After workers finish, only root action statistics are merged:
  visits, means, M2/variance, win-like counts, reveal-frontier coverage, and
  diagnostics. Late-exact evaluation, when eligible, runs once after aggregation
  for the top merged root actions. CLI benchmark commands can override these
  controls with `--root-parallel`, `--root-workers`, `--worker-sim-budget`, and
  `--worker-seed-stride`; those flags are runtime overlays on top of presets,
  not separate planner behavior.

This reuse is intentionally not a large persistent belief-state transposition
table. It is bounded root/near-root metadata for session continuation and
debugging.

Still deferred: large belief-state transposition tables, full persistent search
trees across real turns, neural policy priors, shared-tree concurrency, and
stronger branch-and-bound pruning inside late-exact.
