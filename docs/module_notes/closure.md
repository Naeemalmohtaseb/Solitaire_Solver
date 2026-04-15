# closure

Owns deterministic closure and corridor compression over generated macro moves.

The closure engine is intentionally conservative. It is not a search algorithm
and does not choose among meaningful alternatives. It applies only:

- terminal recognition
- exactly one legal macro move
- Ace/Two foundation advances under a narrow safety rule
- exactly one legal king-headed empty-column placement

Every applied step goes through the move engine, so reveal side effects and
state validation use the same contracts as ordinary move application. Each step
is recorded in a structured transcript with the selected macro move, semantic
tags, reveal information, and the reason closure was allowed to apply it.

Prompt 4 deliberately stops at stock pivots instead of trying to be clever about
draw-cycle decisions. Later deterministic solver work can use the stop reason
and transcript as a compact corridor boundary.
