# session

Owns real-game persistence, replay logs, reveal history, diagnostics export, and
future search subtree reuse metadata.

The v1 session format is versioned JSON (`session-json-v1`). A `SessionRecord`
stores:

- metadata: session id, schema version, engine version, creation timestamp,
  label, preset name, and backend when known
- initial snapshot
- current snapshot
- replayable move steps
- reveal history
- optional lightweight planner continuation metadata

Each snapshot stores the visible state, belief state, and optionally the true
full deterministic state. Real phone-assisted sessions can omit the full state.
Autoplay/debug sessions keep it, which lets replay validate the public belief
against the true hidden assignment state.

Replay starts from the initial snapshot and reapplies stored macro moves through
the existing move engine. Reveal observations are fed through the belief
transition layer; they are not reimplemented here. Replay returns a
`ReplayResult` with a final reconstructed snapshot and mismatch diagnostics.

Library entry points:

- `save_session(path, &record)`
- `load_session(path)`
- `replay_session(&record)`
- `SessionRecord::from_belief(...)`
- `SessionRecord::from_full_state(...)`
- `SessionRecord::from_autoplay_result(...)`
- `SessionRecord::append_observed_move(...)`
- `SessionRecord::set_planner_continuation(...)`
- `SessionRecord::planner_continuation()`

CLI examples:

```powershell
cargo run -p solitaire-cli -- session save --preset fast_benchmark --seed 100 --max-steps 20 --label phone-test --out sessions/game-100.json
cargo run -p solitaire-cli -- session summary --path sessions/game-100.json
cargo run -p solitaire-cli -- session inspect --path sessions/game-100.json
cargo run -p solitaire-cli -- session replay --path sessions/game-100.json
```

Planner reuse now stores bounded root/near-root metadata from the belief planner:
root belief identity, candidate actions, action statistics, deterministic child
keys, reveal-frontier child keys, and a planner/config fingerprint. On the next
turn the planner can recognize a followed move, an observed reveal child, or an
exact cached root. If the user deviates and no exact belief identity matches,
reuse falls back cleanly to a fresh root recommendation. This is lightweight
continuation metadata, not a giant persistent shared search tree.
