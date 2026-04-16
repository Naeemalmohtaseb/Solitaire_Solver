# belief

Owns hidden-information transitions, reveal events, exact reveal chance-node expansion, and uniform hidden-tableau world sampling.

This module should not invent heuristic posterior weighting. The hidden tableau posterior remains exactly uniform over consistent assignments.

Stock/waste order is fully known and deterministic; hidden uncertainty exists only in face-down tableau cards.

The core belief-state type stores a `VisibleState` plus `UnseenCardSet`. It does not store a separate hidden-count array.

Current APIs:

- non-reveal belief transitions apply visible moves exactly and leave `UnseenCardSet` unchanged
- reveal transitions expand an explicit `RevealFrontier`, one child per unseen card
- every reveal child has equal probability and removes exactly the observed card
- `WorldSampler` uniformly shuffles unseen cards over deterministic hidden slots to produce `FullState` determinizations

This module deliberately stops before UCT/MCTS, root action selection, confidence budgeting, or late-game assignment enumeration.
