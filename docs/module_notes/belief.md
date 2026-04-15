# belief

Owns hidden-information transitions, reveal events, exact reveal chance-node expansion, and uniform hidden-tableau world sampling.

This module should not invent heuristic posterior weighting. The hidden tableau posterior remains exactly uniform over consistent assignments.

Stock/waste order is fully known and deterministic; hidden uncertainty exists only in face-down tableau cards.

The core belief-state type stores a `VisibleState` plus `UnseenCardSet`. It does not store a separate hidden-count array.
