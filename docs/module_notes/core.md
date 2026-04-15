# core

Owns the domain state types: visible state, full deterministic state, belief state, tableau columns, and core invariants.

Tableau columns are the single source of truth for per-column hidden-card counts. Belief state carries the unseen card identities and derives hidden counts from the visible tableau.

Unseen tableau cards are represented by a compact `UnseenCardSet` bitmask. Full deterministic worlds use ordered `HiddenSlot` plus `HiddenAssignments` entries so late exact-assignment search has stable positions to enumerate.

Future work should keep search logic out of this module. Validation belongs here; solving does not.
