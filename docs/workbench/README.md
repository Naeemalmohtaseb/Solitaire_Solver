# Solitaire Workbench

Solitaire Workbench is a local desktop launcher for the Draw-3 Klondike solver.
It starts a loopback-only Rust backend, opens an embedded web UI in the default
browser, and calls `solver_core` APIs directly. It does not shell out through
the CLI for normal UI actions.

## Build and launch

Development launch:

```powershell
cargo run -p solitaire-workbench
```

Release build:

```powershell
cargo build --release -p solitaire-workbench
```

On Windows, double-click:

```text
target\release\solitaire-workbench.exe
```

The app creates these default local folders if they do not already exist:

- `sessions/`
- `reports/`
- `data/`
- `models/`
- `regression/`
- `oracle/`

## Architecture

The workbench binary uses a small std-only HTTP server bound to `127.0.0.1`.
Static HTML/CSS/JS assets are embedded into the executable. The frontend calls
structured JSON endpoints such as:

- `/api/session/parse`
- `/api/session/load_path`
- `/api/session/save_path`
- `/api/game/generate`
- `/api/game/generate_autoplay`
- `/api/recommend`
- `/api/solve/one_step`
- `/api/solve/run_to_end`
- `/api/autoplay/run`
- `/api/benchmark/start`
- `/api/task/{id}`

Long benchmark requests run as background tasks. The UI polls task snapshots and
renders progress events emitted by the experiment progress reporter.

## V1 tabs

- **Replay:** load a session JSON, render tableau/foundations/stock/waste, and
  step first/previous/next/last through replay states.
- **Solve:** generate a fresh seeded game, request a recommendation for the
  loaded session, apply one solver step when the session has a true full state,
  or run autoplay to the end from a debug/autoplay full state.
- **Benchmark:** run autoplay, compare, repeated compare, and compare-presets
  tasks with live progress and JSON/CSV downloads.
- **Diagnostics:** audit a loaded/generated session for replay consistency,
  reveal handling, terminal shape, legal-move availability, and suspicious
  step-level flags.
- **Settings:** choose preset/backend, leaf mode, late-exact toggle, V-Net model
  path, and root-parallel overrides.

The shell anticipates Regression, Oracle, and Datasets / ML tabs, but those
remain CLI/library-first in this v1 workbench.

## New games

Use **New Game** in the top bar or the **Game Setup** panel on the Solve tab.
The seed is a deterministic deal identifier: the same seed and preset produce
the same starting board. **Generate Game** creates a saveable debug session from
that deal and loads it into Replay/Solve immediately. **Generate + Autoplay**
creates the same deal, runs the selected backend/preset up to the configured
step limit, and loads the resulting replay trace.

Generated sessions carry provenance metadata, including seed, preset, backend,
step limit, and hidden-card count. The normal **Save Session** action works
after generation, so generated games and autoplay traces can be persisted and
replayed later.

## Diagnostics

Open the **Diagnostics** tab and click **Analyze Current Session** after loading
or generating a session. The report distinguishes replay/state bugs from likely
weak play by checking:

- replay reconstruction against the saved current state
- belief/full-state consistency when true full state is available
- reveal observations against full-state replay
- stock and belief structural validation
- legal move counts before each recorded step and at the final state

Suspicion badges include `BeliefFullMismatch`, `IllegalTerminal`,
`NoMoveButLegalMovesExist`, `RevealUpdateMismatch`, `StockStateSuspicious`,
`ChosenMoveClearlyInferior`, `UnexpectedZeroLegalMoves`, and `ReplayMismatch`.
Clicking a diagnostic step jumps the Replay tab to the corresponding board
position. For now, local action audits use persisted planner snapshots; complete
alternative ranking is shown only when that information was saved with the
session or can be recomputed from the current settings. The action audit also
shows the deterministic strategic score components used for ordering, including
reveal bonus, safe-foundation bonus, stock-access bonus, empty-column bonus, and
churn/foundation-retreat penalties.

## Deferred

- Manual board entry and drag/drop editing
- Native OS file dialogs beyond browser file picker/download behavior
- Live charts
- Regression/Oracle/Dataset workflow tabs
- Neural training/inference changes
- New solver or planner algorithms
