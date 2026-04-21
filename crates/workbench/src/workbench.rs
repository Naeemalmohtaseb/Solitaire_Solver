//! Workbench-facing command layer over `solver_core`.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use solver_core::moves::{apply_atomic_move_full_state, generate_legal_macro_moves, MacroMove};
use solver_core::{
    apply_observed_belief_move, experiment_preset_by_name, load_session, play_game_with_planner,
    recommend_move_belief_uct, recommend_move_pimc_with_vnet, replay_session,
    run_autoplay_benchmark_with_progress, run_autoplay_paired_comparison_with_progress,
    run_autoplay_repeated_comparison_with_progress, save_session, strategic_move_score,
    BenchmarkSuite, Card, DealSeed, ExperimentPreset, ExperimentRunner, LeafEvaluationMode,
    PlannerBackend, ProgressEvent, ProgressReporter, RootParallelConfigOverride, SessionMetadata,
    SessionRecord, StrategicMoveScore, Suit, VNetInferenceConfig, EXPERIMENT_PRESET_NAMES, VERSION,
};

/// Shared asynchronous benchmark task registry.
#[derive(Debug, Default)]
pub struct TaskStore {
    next_id: u64,
    tasks: HashMap<String, TaskState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskState {
    pub id: String,
    pub kind: String,
    pub progress: Option<ProgressEvent>,
    pub done: bool,
    pub result: Option<Value>,
    pub error: Option<String>,
}

/// Lightweight health/status payload for the UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkbenchStatus {
    pub app_name: String,
    pub version: String,
    pub backend_status: String,
    pub default_folders: Vec<String>,
}

/// Returns current backend status.
pub fn status() -> WorkbenchStatus {
    WorkbenchStatus {
        app_name: "Solitaire Workbench".to_string(),
        version: VERSION.to_string(),
        backend_status: "solver_core loaded".to_string(),
        default_folders: [
            "sessions",
            "reports",
            "data",
            "models",
            "regression",
            "oracle",
        ]
        .iter()
        .map(|value| value.to_string())
        .collect(),
    }
}

/// Returns known experiment preset names.
pub fn presets() -> Vec<String> {
    EXPERIMENT_PRESET_NAMES
        .iter()
        .map(|name| name.to_string())
        .collect()
}

/// UI-ready board snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiBoardState {
    pub foundations: Vec<Option<String>>,
    pub tableau: Vec<UiTableauColumn>,
    pub stock: UiStockState,
    pub hidden_count: usize,
    pub structural_win: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiTableauColumn {
    pub index: usize,
    pub hidden_count: u8,
    pub face_up: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiStockState {
    pub stock_len: usize,
    pub waste_len: usize,
    pub accessible_card: Option<String>,
    pub pass_index: u32,
    pub draw_count: u8,
    pub ring_len: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiReplayTrace {
    pub label: String,
    pub metadata: UiSessionMetadata,
    pub steps: Vec<UiReplayStep>,
    pub termination: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiSessionMetadata {
    pub session_id: String,
    pub label: Option<String>,
    pub preset_name: Option<String>,
    pub backend: Option<String>,
    pub schema_version: String,
    pub engine_version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiReplayStep {
    pub index: usize,
    pub board: UiBoardState,
    pub move_text: Option<String>,
    pub revealed_card: Option<String>,
    pub backend: Option<String>,
    pub preset: Option<String>,
    pub planner: Option<UiPlannerSnapshot>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiPlannerSnapshot {
    pub best_value: f64,
    pub elapsed_ms: u64,
    pub deterministic_nodes: u64,
    pub root_visits: u64,
    pub late_exact_triggered: bool,
    pub root_parallel_used: bool,
    pub root_parallel_workers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionParseRequest {
    pub contents: String,
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathRequest {
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveSessionPathRequest {
    pub path: String,
    pub session: SessionRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionResponse {
    pub session: SessionRecord,
    pub replay: UiReplayTrace,
    pub generated: Option<GeneratedGameMetadata>,
}

/// Metadata attached to workbench-created games.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedGameMetadata {
    pub seed: u64,
    pub preset: String,
    pub backend: String,
    pub max_steps: Option<usize>,
    pub kind: GeneratedGameKind,
    pub hidden_count: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeneratedGameKind {
    GeneratedOnly,
    GeneratedAutoplay,
}

/// Parses a session JSON document from the UI file picker.
pub fn parse_session(body: &[u8]) -> Result<SessionResponse, String> {
    let request: SessionParseRequest = parse_json(body)?;
    let mut session: SessionRecord = serde_json::from_str(&request.contents)
        .map_err(|err| format!("invalid session JSON: {err}"))?;
    if let Some(label) = request.label {
        session.metadata.label = Some(label);
    }
    session
        .validate_structure()
        .map_err(|err| format!("invalid session: {err}"))?;
    let replay = replay_trace_from_session(&session)?;
    Ok(SessionResponse {
        session,
        replay,
        generated: None,
    })
}

/// Loads a session from a local path typed into the workbench.
pub fn load_session_from_path(body: &[u8]) -> Result<SessionResponse, String> {
    let request: PathRequest = parse_json(body)?;
    let session = load_session(PathBuf::from(request.path)).map_err(|err| err.to_string())?;
    let replay = replay_trace_from_session(&session)?;
    Ok(SessionResponse {
        session,
        replay,
        generated: None,
    })
}

/// Saves a session to a local path typed into the workbench.
pub fn save_session_to_path(body: &[u8]) -> Result<UiSessionMetadata, String> {
    let request: SaveSessionPathRequest = parse_json(body)?;
    save_session(PathBuf::from(request.path), &request.session).map_err(|err| err.to_string())?;
    Ok(session_metadata(&request.session))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverSettingsRequest {
    pub preset: String,
    pub backend: Option<String>,
    pub leaf_eval_mode: Option<String>,
    pub vnet_model_path: Option<String>,
    pub max_steps: Option<usize>,
    pub late_exact_enabled: Option<bool>,
    pub root_parallel: Option<bool>,
    pub root_workers: Option<usize>,
    pub worker_sim_budget: Option<usize>,
    pub worker_seed_stride: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendRequest {
    pub session: SessionRecord,
    pub settings: SolverSettingsRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticRequest {
    pub session: SessionRecord,
    pub settings: Option<SolverSettingsRequest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiDiagnosticReport {
    pub overall_status: String,
    pub consistency_held: bool,
    pub replay_matched: bool,
    pub terminal_audit: UiTerminalAudit,
    pub total_suspicious_flags: usize,
    pub most_severe_issue: Option<String>,
    pub final_termination_reason: String,
    pub steps: Vec<UiDiagnosticStep>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiTerminalAudit {
    pub structural_win: bool,
    pub legal_moves_remaining: usize,
    pub terminal_status_valid: bool,
    pub likely_bug: bool,
    pub likely_weak_decision_making: bool,
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiDiagnosticStep {
    pub step_index: usize,
    pub chosen_move: String,
    pub revealed_card: Option<String>,
    pub hidden_count: usize,
    pub legal_move_count: usize,
    pub consistency_ok: bool,
    pub severity: DiagnosticSeverity,
    pub flags: Vec<UiDiagnosticFlag>,
    pub action_audit: Option<UiActionAudit>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Ok,
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UiDiagnosticFlag {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiActionAudit {
    pub chosen_move: String,
    pub chosen_planner_value: Option<f64>,
    pub top_alternatives: Vec<UiActionAlternative>,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendResponse {
    pub best_move: Option<String>,
    pub best_value: f64,
    pub elapsed_ms: u64,
    pub deterministic_nodes: u64,
    pub root_visits: u64,
    pub late_exact_triggered: bool,
    pub root_parallel_used: bool,
    pub alternatives: Vec<UiActionAlternative>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiActionAlternative {
    pub move_text: String,
    pub visits: usize,
    pub mean_value: f64,
    pub stderr: f64,
}

/// Recommends the next move for the current session belief state.
pub fn recommend_move(body: &[u8]) -> Result<RecommendResponse, String> {
    let request: RecommendRequest = parse_json(body)?;
    recommend_for_session(&request.session, &request.settings)
}

/// Audits a saved/generated session for replay, reveal, and terminal consistency.
pub fn analyze_session(body: &[u8]) -> Result<UiDiagnosticReport, String> {
    let request: DiagnosticRequest = parse_json(body)?;
    diagnostic_report_for_session(&request.session, request.settings.as_ref())
}

/// Alias for callers that name the loaded workbench state as "current".
pub fn analyze_current(body: &[u8]) -> Result<UiDiagnosticReport, String> {
    analyze_session(body)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolveStepResponse {
    pub session: SessionRecord,
    pub replay: UiReplayTrace,
    pub recommendation: RecommendResponse,
}

/// Recommends and applies one move when the session contains a true full state.
pub fn solve_one_step(body: &[u8]) -> Result<SolveStepResponse, String> {
    let request: RecommendRequest = parse_json(body)?;
    let recommendation = recommend_for_session(&request.session, &request.settings)?;
    let best = best_macro_move(&request.session, &request.settings)?
        .ok_or_else(|| "no legal move is available".to_string())?;

    let mut session = request.session;
    if session.current_snapshot.full_state.is_none() {
        return Err(
            "Solve One Step needs a debug/autoplay session with true full state".to_string(),
        );
    }
    session
        .append_observed_move(best, None, None)
        .map_err(|err| err.to_string())?;
    let replay = replay_trace_from_session(&session)?;
    Ok(SolveStepResponse {
        session,
        replay,
        recommendation,
    })
}

/// Runs autoplay from the current true full state when available.
pub fn solve_run_to_end(body: &[u8]) -> Result<Value, String> {
    let request: RecommendRequest = parse_json(body)?;
    let full_state = request
        .session
        .current_snapshot
        .full_state
        .as_ref()
        .ok_or_else(|| {
            "Run To End needs a debug/autoplay session with true full state".to_string()
        })?;
    let preset = configured_preset(&request.settings)?;
    let mut autoplay = preset.autoplay;
    if let Some(max_steps) = request.settings.max_steps {
        autoplay.max_steps = max_steps;
    }
    let result = play_game_with_planner(full_state, &preset.solver, &autoplay)
        .map_err(|err| err.to_string())?;
    serde_json::to_value(result).map_err(|err| err.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoplayRunRequest {
    pub settings: SolverSettingsRequest,
    pub seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateGameRequest {
    pub settings: SolverSettingsRequest,
    pub seed: u64,
}

/// Generates a fresh seeded game and returns it as a saveable session.
pub fn generate_game(body: &[u8]) -> Result<SessionResponse, String> {
    let request: GenerateGameRequest = parse_json(body)?;
    validate_optional_vnet_path(&request.settings)?;
    let preset = configured_preset(&request.settings)?;
    let deal = ExperimentRunner
        .generate_deal(DealSeed(request.seed))
        .map_err(|err| format!("generation failed: {err}"))?;
    let generated = generated_metadata(
        request.seed,
        &preset,
        request.settings.max_steps,
        GeneratedGameKind::GeneratedOnly,
        deal.belief_state.hidden_card_count(),
    );
    let session = SessionRecord::from_full_state(
        SessionMetadata::generated(Some(format!("Generated seed {}", request.seed)))
            .with_solver_provenance(Some(preset.name.clone()), Some(preset.autoplay.backend)),
        deal.full_state,
    )
    .map_err(|err| format!("generation failed: {err}"))?;
    let replay = replay_trace_from_session(&session)?;
    Ok(SessionResponse {
        session,
        replay,
        generated: Some(generated),
    })
}

/// Generates a fresh seeded game, autoplays it, and returns a replayable session.
pub fn generate_autoplay(body: &[u8]) -> Result<SessionResponse, String> {
    let request: GenerateGameRequest = parse_json(body)?;
    validate_optional_vnet_path(&request.settings)?;
    let preset = configured_preset(&request.settings)?;
    let deal = ExperimentRunner
        .generate_deal(DealSeed(request.seed))
        .map_err(|err| format!("generation failed: {err}"))?;
    let result = play_game_with_planner(&deal.full_state, &preset.solver, &preset.autoplay)
        .map_err(|err| format!("autoplay failed: {err}"))?;
    let session = SessionRecord::from_autoplay_result(
        SessionMetadata::generated(Some(format!("Autoplay seed {}", request.seed)))
            .with_solver_provenance(Some(preset.name.clone()), Some(preset.autoplay.backend)),
        deal.full_state,
        &result,
    )
    .map_err(|err| format!("autoplay session reconstruction failed: {err}"))?;
    let hidden_count = session.current_snapshot.belief.hidden_card_count();
    let replay = replay_trace_from_session(&session)?;
    Ok(SessionResponse {
        session,
        replay,
        generated: Some(generated_metadata(
            request.seed,
            &preset,
            request.settings.max_steps,
            GeneratedGameKind::GeneratedAutoplay,
            hidden_count,
        )),
    })
}

/// Runs one autoplay game from a seeded deal.
pub fn run_autoplay(body: &[u8]) -> Result<Value, String> {
    let request: AutoplayRunRequest = parse_json(body)?;
    let preset = configured_preset(&request.settings)?;
    let deal = ExperimentRunner
        .generate_deal(solver_core::DealSeed(request.seed))
        .map_err(|err| err.to_string())?;
    let result = play_game_with_planner(&deal.full_state, &preset.solver, &preset.autoplay)
        .map_err(|err| err.to_string())?;
    serde_json::to_value(result).map_err(|err| err.to_string())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStartRequest {
    pub benchmark_type: String,
    pub preset: Option<String>,
    pub baseline: Option<String>,
    pub candidate: Option<String>,
    pub presets: Option<Vec<String>>,
    pub games: usize,
    pub repetitions: Option<usize>,
    pub seed: u64,
    pub rank_by: Option<String>,
    pub max_steps: Option<usize>,
    pub settings: SolverSettingsRequest,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStartResponse {
    pub task_id: String,
}

/// Starts a background benchmark task and returns its task id.
pub fn start_benchmark_task(
    tasks: &Arc<Mutex<TaskStore>>,
    body: &[u8],
) -> Result<TaskStartResponse, String> {
    let request: BenchmarkStartRequest = parse_json(body)?;
    let task_id = {
        let mut tasks = tasks
            .lock()
            .map_err(|_| "task store poisoned".to_string())?;
        tasks.next_id += 1;
        let id = format!("task-{}", tasks.next_id);
        tasks.tasks.insert(
            id.clone(),
            TaskState {
                id: id.clone(),
                kind: request.benchmark_type.clone(),
                progress: None,
                done: false,
                result: None,
                error: None,
            },
        );
        id
    };

    let task_id_for_thread = task_id.clone();
    let tasks_for_thread = Arc::clone(tasks);
    thread::spawn(move || {
        let result = run_benchmark_request(request, &tasks_for_thread, &task_id_for_thread);
        let mut tasks = match tasks_for_thread.lock() {
            Ok(tasks) => tasks,
            Err(_) => return,
        };
        if let Some(task) = tasks.tasks.get_mut(&task_id_for_thread) {
            task.done = true;
            match result {
                Ok(value) => task.result = Some(value),
                Err(err) => task.error = Some(err),
            }
        }
    });

    Ok(TaskStartResponse { task_id })
}

/// Returns a background task snapshot.
pub fn task_snapshot(tasks: &Arc<Mutex<TaskStore>>, task_id: &str) -> Result<TaskState, String> {
    let tasks = tasks
        .lock()
        .map_err(|_| "task store poisoned".to_string())?;
    tasks
        .tasks
        .get(task_id)
        .cloned()
        .ok_or_else(|| format!("unknown task id {task_id}"))
}

fn run_benchmark_request(
    request: BenchmarkStartRequest,
    tasks: &Arc<Mutex<TaskStore>>,
    task_id: &str,
) -> Result<Value, String> {
    let mut reporter = TaskProgressReporter {
        tasks: Arc::clone(tasks),
        task_id: task_id.to_string(),
    };
    match request.benchmark_type.as_str() {
        "autoplay" => {
            let preset_name = request
                .preset
                .as_deref()
                .unwrap_or(&request.settings.preset);
            let mut preset = configured_preset(&SolverSettingsRequest {
                preset: preset_name.to_string(),
                ..request.settings.clone()
            })?;
            if let Some(max_steps) = request.max_steps {
                preset.autoplay.max_steps = max_steps;
            }
            let suite = BenchmarkSuite::from_base_seed(
                format!("workbench-{preset_name}"),
                request.seed,
                request.games,
            );
            let result = run_autoplay_benchmark_with_progress(
                &suite,
                &preset.autoplay_benchmark_config(),
                &mut reporter,
            )
            .map_err(|err| err.to_string())?;
            serde_json::to_value(result).map_err(|err| err.to_string())
        }
        "compare" => {
            let baseline_name = request
                .baseline
                .as_deref()
                .ok_or_else(|| "baseline preset is required".to_string())?;
            let candidate_name = request
                .candidate
                .as_deref()
                .ok_or_else(|| "candidate preset is required".to_string())?;
            let mut baseline = configured_preset(&SolverSettingsRequest {
                preset: baseline_name.to_string(),
                ..request.settings.clone()
            })?;
            let mut candidate = configured_preset(&SolverSettingsRequest {
                preset: candidate_name.to_string(),
                ..request.settings.clone()
            })?;
            if let Some(max_steps) = request.max_steps {
                baseline.autoplay.max_steps = max_steps;
                candidate.autoplay.max_steps = max_steps;
            }
            let suite =
                BenchmarkSuite::from_base_seed("workbench-compare", request.seed, request.games);
            let result = run_autoplay_paired_comparison_with_progress(
                &suite,
                &baseline.autoplay_benchmark_config(),
                &candidate.autoplay_benchmark_config(),
                &mut reporter,
            )
            .map_err(|err| err.to_string())?;
            serde_json::to_value(result).map_err(|err| err.to_string())
        }
        "repeated_compare" => {
            let baseline_name = request
                .baseline
                .as_deref()
                .ok_or_else(|| "baseline preset is required".to_string())?;
            let candidate_name = request
                .candidate
                .as_deref()
                .ok_or_else(|| "candidate preset is required".to_string())?;
            let baseline = configured_preset(&SolverSettingsRequest {
                preset: baseline_name.to_string(),
                ..request.settings.clone()
            })?;
            let candidate = configured_preset(&SolverSettingsRequest {
                preset: candidate_name.to_string(),
                ..request.settings.clone()
            })?;
            let result = run_autoplay_repeated_comparison_with_progress(
                "workbench-repeated",
                request.seed,
                request.games,
                request.repetitions.unwrap_or(3),
                &baseline.autoplay_benchmark_config(),
                &candidate.autoplay_benchmark_config(),
                &mut reporter,
            )
            .map_err(|err| err.to_string())?;
            serde_json::to_value(result).map_err(|err| err.to_string())
        }
        "compare_presets" => {
            let names = request.presets.unwrap_or_else(|| {
                EXPERIMENT_PRESET_NAMES
                    .iter()
                    .map(|name| name.to_string())
                    .collect()
            });
            let mut presets = Vec::with_capacity(names.len());
            for name in names {
                presets.push(configured_preset(&SolverSettingsRequest {
                    preset: name,
                    ..request.settings.clone()
                })?);
            }
            if let Some(max_steps) = request.max_steps {
                for preset in &mut presets {
                    preset.autoplay.max_steps = max_steps;
                }
            }
            let suite =
                BenchmarkSuite::from_base_seed("workbench-presets", request.seed, request.games);
            let rank = match request.rank_by.as_deref() {
                Some("win_rate") => solver_core::PresetRankingMetric::WinRate,
                Some("time") => solver_core::PresetRankingMetric::TimePerGame,
                _ => solver_core::PresetRankingMetric::Efficiency,
            };
            let result = solver_core::compare_experiment_presets_on_suite_with_progress(
                &suite,
                &presets,
                rank,
                &mut reporter,
            )
            .map_err(|err| err.to_string())?;
            serde_json::to_value(result).map_err(|err| err.to_string())
        }
        other => Err(format!("unknown benchmark type {other:?}")),
    }
}

struct TaskProgressReporter {
    tasks: Arc<Mutex<TaskStore>>,
    task_id: String,
}

impl ProgressReporter for TaskProgressReporter {
    fn report(&mut self, event: &ProgressEvent) {
        if let Ok(mut tasks) = self.tasks.lock() {
            if let Some(task) = tasks.tasks.get_mut(&self.task_id) {
                task.progress = Some(event.clone());
            }
        }
    }
}

fn recommend_for_session(
    session: &SessionRecord,
    settings: &SolverSettingsRequest,
) -> Result<RecommendResponse, String> {
    let best = recommendation_parts(session, settings)?;
    Ok(best.0)
}

fn diagnostic_report_for_session(
    session: &SessionRecord,
    settings: Option<&SolverSettingsRequest>,
) -> Result<UiDiagnosticReport, String> {
    let replay = replay_session(session).map_err(|err| err.to_string())?;
    let mut belief = session.initial_snapshot.belief.clone();
    let mut full_state = session.initial_snapshot.full_state.clone();
    let mut steps = Vec::with_capacity(session.steps.len());
    let mut all_flags = Vec::<UiDiagnosticFlag>::new();

    for step in &session.steps {
        let mut flags = Vec::<UiDiagnosticFlag>::new();
        let hidden_count = belief.hidden_card_count();
        let legal_moves = generate_legal_macro_moves(&belief.visible);
        let action_audit = action_audit_for_step(&belief, settings, step);
        if legal_moves.is_empty() {
            flags.push(flag(
                "UnexpectedZeroLegalMoves",
                "No legal moves existed before this recorded step.",
            ));
        }
        if !legal_moves
            .iter()
            .any(|candidate| candidate.atomic == step.move_record.macro_move.atomic)
        {
            flags.push(flag(
                "ReplayMismatch",
                "The recorded move was not legal from the replayed public state.",
            ));
        }

        if let Err(error) = belief.visible.stock.validate_structure() {
            flags.push(flag(
                "StockStateSuspicious",
                format!("Stock structure validation failed: {error}"),
            ));
        }
        if let Err(error) = belief.validate_consistency_against_visible() {
            flags.push(flag(
                "BeliefFullMismatch",
                format!("Belief/visible validation failed: {error}"),
            ));
        }
        if let Some(full_state_ref) = &full_state {
            if let Err(error) =
                solver_core::validate_belief_against_full_state(&belief, full_state_ref)
            {
                flags.push(flag(
                    "BeliefFullMismatch",
                    format!("Belief/full-state validation failed before move: {error}"),
                ));
            }
        }

        let reveal_from_full = if let Some(full_state_ref) = &mut full_state {
            match apply_atomic_move_full_state(full_state_ref, step.move_record.macro_move.atomic) {
                Ok(transition) => transition.outcome.revealed,
                Err(error) => {
                    flags.push(flag(
                        "ReplayMismatch",
                        format!("Full-state move application failed: {error}"),
                    ));
                    None
                }
            }
        } else {
            step.reveal_record.as_ref().map(|record| record.reveal)
        };
        let recorded_reveal = step.reveal_record.as_ref().map(|record| record.reveal);
        if recorded_reveal != reveal_from_full {
            flags.push(flag(
                "RevealUpdateMismatch",
                "Recorded reveal did not match replayed full-state reveal.",
            ));
        }

        match apply_observed_belief_move(
            &belief,
            step.move_record.macro_move.atomic,
            reveal_from_full.map(|reveal| reveal.card),
        ) {
            Ok((next_belief, outcome)) => {
                if outcome.revealed != reveal_from_full {
                    flags.push(flag(
                        "RevealUpdateMismatch",
                        "Belief transition reveal did not match replayed reveal.",
                    ));
                }
                belief = next_belief;
                if let Some(full_state_ref) = &full_state {
                    if let Err(error) =
                        solver_core::validate_belief_against_full_state(&belief, full_state_ref)
                    {
                        flags.push(flag(
                            "BeliefFullMismatch",
                            format!("Belief/full-state validation failed after move: {error}"),
                        ));
                    }
                }
            }
            Err(error) => flags.push(flag(
                "ReplayMismatch",
                format!("Belief move replay failed: {error}"),
            )),
        }

        if action_audit
            .as_ref()
            .is_some_and(|audit| audit.note.contains("clearly below"))
        {
            flags.push(flag(
                "ChosenMoveClearlyInferior",
                "Planner alternatives suggest the chosen move was clearly below the best local alternative.",
            ));
        }

        let consistency_ok = !flags.iter().any(|flag| {
            matches!(
                flag.code.as_str(),
                "BeliefFullMismatch" | "RevealUpdateMismatch" | "ReplayMismatch"
            )
        });
        let severity = severity_for_flags(&flags);
        all_flags.extend(flags.iter().cloned());
        steps.push(UiDiagnosticStep {
            step_index: step.step_index,
            chosen_move: format!("{:?}", step.move_record.macro_move.kind),
            revealed_card: reveal_from_full.map(|reveal| card_text(reveal.card)),
            hidden_count,
            legal_move_count: legal_moves.len(),
            consistency_ok,
            severity,
            flags,
            action_audit,
        });
    }

    for mismatch in &replay.mismatches {
        all_flags.push(flag(
            "ReplayMismatch",
            match mismatch.step_index {
                Some(index) => format!("Replay mismatch at step {index}: {}", mismatch.message),
                None => format!("Replay mismatch: {}", mismatch.message),
            },
        ));
    }

    let final_belief = &session.current_snapshot.belief;
    let final_legal_moves = generate_legal_macro_moves(&final_belief.visible).len();
    let terminal_audit =
        terminal_audit_for_state(final_belief, final_legal_moves, replay.matched, &all_flags);

    let consistency_held = replay.matched
        && session.current_snapshot.validate_consistency().is_ok()
        && !all_flags.iter().any(|flag| {
            matches!(
                flag.code.as_str(),
                "BeliefFullMismatch" | "RevealUpdateMismatch" | "ReplayMismatch"
            )
        });
    let most_severe_issue = most_severe_flag(&all_flags).map(|flag| flag.code.clone());
    let final_termination_reason = final_termination_reason(final_belief, final_legal_moves);
    let overall_status = if !consistency_held {
        "Replay or consistency issue detected"
    } else if terminal_audit.likely_bug {
        "Likely bug"
    } else if terminal_audit.likely_weak_decision_making {
        "Likely weak decision-making"
    } else if terminal_audit.structural_win {
        "Won"
    } else if final_legal_moves > 0 {
        "Trace stopped before terminal state"
    } else {
        "No legal moves"
    }
    .to_string();

    Ok(UiDiagnosticReport {
        overall_status,
        consistency_held,
        replay_matched: replay.matched,
        terminal_audit,
        total_suspicious_flags: all_flags.len(),
        most_severe_issue,
        final_termination_reason,
        steps,
    })
}

fn action_audit_for_step(
    belief_before_step: &solver_core::BeliefState,
    settings: Option<&SolverSettingsRequest>,
    step: &solver_core::SessionStep,
) -> Option<UiActionAudit> {
    if let Some(settings) = settings {
        let diagnostic_session = SessionRecord::from_belief(
            SessionMetadata::generated(Some("diagnostic-action-audit".to_string())),
            belief_before_step.clone(),
        )
        .ok()?;
        let recommendation = recommend_for_session(&diagnostic_session, settings).ok()?;
        let chosen_move = format!("{:?}", step.move_record.macro_move.kind);
        let chosen_strategy =
            strategic_move_score(&belief_before_step.visible, &step.move_record.macro_move);
        let chosen_value = recommendation
            .alternatives
            .iter()
            .find(|alternative| alternative.move_text == chosen_move)
            .map(|alternative| alternative.mean_value);
        let note = match (recommendation.best_move.as_deref(), chosen_value) {
            (Some(best), Some(value))
                if best != chosen_move && recommendation.best_value - value > 0.15 =>
            {
                format!(
                    "Fresh local audit puts this move clearly below {best} by {:.3}.",
                    recommendation.best_value - value
                )
            }
            (Some(best), Some(_)) if best != chosen_move => {
                format!("Fresh local audit prefers {best}, but the value gap is not large.")
            }
            (Some(_), Some(_)) => "Fresh local audit agrees with the recorded choice.".to_string(),
            (Some(best), None) => {
                format!("Fresh local audit prefers {best}; the recorded move was not found in returned alternatives.")
            }
            (None, _) => "Fresh local audit found no legal recommendation.".to_string(),
        };
        let note = format!("{note} {}", strategic_score_note(chosen_strategy));
        return Some(UiActionAudit {
            chosen_move,
            chosen_planner_value: chosen_value.or_else(|| {
                step.move_record
                    .planner
                    .as_ref()
                    .map(|planner| planner.best_value)
            }),
            top_alternatives: recommendation.alternatives.into_iter().take(5).collect(),
            note,
        });
    }

    let planner = step.move_record.planner.as_ref()?;
    let chosen_strategy =
        strategic_move_score(&belief_before_step.visible, &step.move_record.macro_move);
    Some(UiActionAudit {
        chosen_move: format!("{:?}", step.move_record.macro_move.kind),
        chosen_planner_value: Some(planner.best_value),
        top_alternatives: Vec::new(),
        note: format!(
            "Stored planner snapshot is available; full alternative ranking was not persisted in this session. {}",
            strategic_score_note(chosen_strategy)
        ),
    })
}

fn strategic_score_note(score: StrategicMoveScore) -> String {
    format!(
        "Strategic score {} [reveal +{}, safe foundation +{}, foundation +{}, stock +{}, empty +{}, unblock +{}, churn {}, retreat {}].",
        score.total,
        score.reveal_bonus,
        score.safe_foundation_bonus,
        score.foundation_progress_bonus,
        score.stock_access_bonus,
        score.empty_column_bonus,
        score.unblock_bonus,
        score.churn_penalty,
        score.foundation_retreat_penalty
    )
}

fn terminal_audit_for_state(
    belief: &solver_core::BeliefState,
    legal_moves_remaining: usize,
    replay_matched: bool,
    flags: &[UiDiagnosticFlag],
) -> UiTerminalAudit {
    let structural_win = belief.visible.is_structural_win();
    let consistency_problem = flags.iter().any(|flag| {
        matches!(
            flag.code.as_str(),
            "BeliefFullMismatch" | "RevealUpdateMismatch" | "ReplayMismatch"
        )
    });
    let terminal_status_valid = structural_win || legal_moves_remaining == 0;
    let likely_bug = !replay_matched || consistency_problem;
    let likely_weak_decision_making = !likely_bug && !structural_win && legal_moves_remaining == 0;
    let note = if structural_win {
        "Final state is structurally won.".to_string()
    } else if legal_moves_remaining == 0 && likely_bug {
        "No legal moves remain, but replay or consistency issues should be inspected first."
            .to_string()
    } else if legal_moves_remaining == 0 {
        "No legal moves remain and replay consistency held; this looks like weak play rather than a state bug."
            .to_string()
    } else {
        "The trace stopped with legal moves still available; this usually indicates a step/budget limit rather than a terminal no-move loss."
            .to_string()
    };
    UiTerminalAudit {
        structural_win,
        legal_moves_remaining,
        terminal_status_valid,
        likely_bug,
        likely_weak_decision_making,
        note,
    }
}

fn final_termination_reason(belief: &solver_core::BeliefState, legal_moves: usize) -> String {
    if belief.visible.is_structural_win() {
        "StructuralWin".to_string()
    } else if legal_moves == 0 {
        "NoLegalMove".to_string()
    } else {
        "StoppedWithLegalMovesRemaining".to_string()
    }
}

fn flag(code: impl Into<String>, message: impl Into<String>) -> UiDiagnosticFlag {
    UiDiagnosticFlag {
        code: code.into(),
        message: message.into(),
    }
}

fn severity_for_flags(flags: &[UiDiagnosticFlag]) -> DiagnosticSeverity {
    if flags.iter().any(|flag| {
        matches!(
            flag.code.as_str(),
            "BeliefFullMismatch" | "RevealUpdateMismatch" | "ReplayMismatch" | "IllegalTerminal"
        )
    }) {
        DiagnosticSeverity::Error
    } else if flags.iter().any(|flag| {
        matches!(
            flag.code.as_str(),
            "ChosenMoveClearlyInferior"
                | "NoMoveButLegalMovesExist"
                | "StockStateSuspicious"
                | "UnexpectedZeroLegalMoves"
        )
    }) {
        DiagnosticSeverity::Warning
    } else {
        DiagnosticSeverity::Ok
    }
}

fn most_severe_flag(flags: &[UiDiagnosticFlag]) -> Option<&UiDiagnosticFlag> {
    flags.iter().max_by_key(|flag| match flag.code.as_str() {
        "BeliefFullMismatch" | "RevealUpdateMismatch" | "ReplayMismatch" | "IllegalTerminal" => 3,
        "NoMoveButLegalMovesExist"
        | "ChosenMoveClearlyInferior"
        | "StockStateSuspicious"
        | "UnexpectedZeroLegalMoves" => 2,
        _ => 1,
    })
}

fn best_macro_move(
    session: &SessionRecord,
    settings: &SolverSettingsRequest,
) -> Result<Option<MacroMove>, String> {
    Ok(recommendation_parts(session, settings)?.1)
}

fn recommendation_parts(
    session: &SessionRecord,
    settings: &SolverSettingsRequest,
) -> Result<(RecommendResponse, Option<MacroMove>), String> {
    let preset = configured_preset(settings)?;
    let belief = &session.current_snapshot.belief;
    let backend = settings
        .backend
        .as_deref()
        .and_then(parse_backend)
        .unwrap_or(preset.autoplay.backend);

    match backend {
        PlannerBackend::Pimc => {
            let root_config = preset.root_benchmark_config();
            let rec = recommend_move_pimc_with_vnet(
                belief,
                root_config.deterministic,
                preset.autoplay.pimc,
                preset.solver.deterministic.vnet_inference.clone(),
            )
            .map_err(|err| err.to_string())?;
            let alternatives = rec
                .action_stats
                .iter()
                .map(|stats| UiActionAlternative {
                    move_text: format!("{:?}", stats.action.kind),
                    visits: stats.visits,
                    mean_value: stats.mean_value,
                    stderr: stats.standard_error,
                })
                .collect();
            Ok((
                RecommendResponse {
                    best_move: rec.best_move.as_ref().map(|m| format!("{:?}", m.kind)),
                    best_value: rec.best_value,
                    elapsed_ms: rec.elapsed_ms,
                    deterministic_nodes: rec.deterministic_nodes,
                    root_visits: rec.sample_count as u64,
                    late_exact_triggered: false,
                    root_parallel_used: false,
                    alternatives,
                },
                rec.best_move,
            ))
        }
        PlannerBackend::BeliefUct | PlannerBackend::BeliefUctLateExact => {
            let mut solver = preset.solver.clone();
            solver.late_exact.enabled = backend == PlannerBackend::BeliefUctLateExact;
            let rec = recommend_move_belief_uct(belief, &solver, &solver.belief_planner)
                .map_err(|err| err.to_string())?;
            let alternatives = rec
                .action_stats
                .iter()
                .map(|stats| UiActionAlternative {
                    move_text: format!("{:?}", stats.action.kind),
                    visits: stats.visits,
                    mean_value: stats.mean_value,
                    stderr: stats.standard_error,
                })
                .collect();
            Ok((
                RecommendResponse {
                    best_move: rec.best_move.as_ref().map(|m| format!("{:?}", m.kind)),
                    best_value: rec.best_value,
                    elapsed_ms: rec.elapsed_ms,
                    deterministic_nodes: rec.deterministic_nodes
                        + rec.late_exact_deterministic_nodes,
                    root_visits: rec.simulations_run as u64,
                    late_exact_triggered: rec.late_exact_triggered,
                    root_parallel_used: rec.root_parallel_used,
                    alternatives,
                },
                rec.best_move,
            ))
        }
    }
}

fn generated_metadata(
    seed: u64,
    preset: &ExperimentPreset,
    max_steps: Option<usize>,
    kind: GeneratedGameKind,
    hidden_count: usize,
) -> GeneratedGameMetadata {
    GeneratedGameMetadata {
        seed,
        preset: preset.name.clone(),
        backend: format!("{:?}", preset.autoplay.backend),
        max_steps,
        kind,
        hidden_count,
    }
}

fn validate_optional_vnet_path(settings: &SolverSettingsRequest) -> Result<(), String> {
    let Some(path) = settings
        .vnet_model_path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
    else {
        return Ok(());
    };
    if PathBuf::from(path).exists() {
        Ok(())
    } else {
        Err(format!("invalid V-Net model path: {path}"))
    }
}

fn configured_preset(settings: &SolverSettingsRequest) -> Result<ExperimentPreset, String> {
    let mut preset = experiment_preset_by_name(&settings.preset)
        .ok_or_else(|| format!("unknown preset {:?}", settings.preset))?;
    if let Some(mode) = settings.leaf_eval_mode.as_deref().and_then(parse_leaf_mode) {
        preset.solver.deterministic.leaf_eval_mode = mode;
    }
    if let Some(path) = &settings.vnet_model_path {
        if !path.trim().is_empty() {
            preset.solver.deterministic.leaf_eval_mode = LeafEvaluationMode::VNet;
            preset.solver.deterministic.vnet_inference = VNetInferenceConfig {
                enable_vnet: true,
                backend: Default::default(),
                model_path: Some(PathBuf::from(path)),
                fallback_to_heuristic: true,
                batch_leaf_eval: false,
            };
        }
    }
    if let Some(enabled) = settings.late_exact_enabled {
        preset.solver.late_exact.enabled = enabled;
    }
    let overrides = RootParallelConfigOverride {
        enable_root_parallel: settings.root_parallel,
        root_workers: settings.root_workers,
        worker_simulation_budget: settings.worker_sim_budget,
        worker_seed_stride: settings.worker_seed_stride,
    };
    overrides
        .apply_to_solver_config(&mut preset.solver)
        .map_err(|err| err.to_string())?;
    if let Some(backend) = settings.backend.as_deref().and_then(parse_backend) {
        preset.autoplay.backend = backend;
    }
    if let Some(max_steps) = settings.max_steps {
        preset.autoplay.max_steps = max_steps;
    }
    Ok(preset)
}

fn parse_backend(value: &str) -> Option<PlannerBackend> {
    match value {
        "pimc" | "Pimc" => Some(PlannerBackend::Pimc),
        "belief_uct" | "BeliefUct" => Some(PlannerBackend::BeliefUct),
        "belief_uct_late_exact" | "BeliefUctLateExact" => Some(PlannerBackend::BeliefUctLateExact),
        _ => None,
    }
}

fn parse_leaf_mode(value: &str) -> Option<LeafEvaluationMode> {
    match value {
        "heuristic" | "Heuristic" => Some(LeafEvaluationMode::Heuristic),
        "vnet" | "VNet" => Some(LeafEvaluationMode::VNet),
        _ => None,
    }
}

fn replay_trace_from_session(session: &SessionRecord) -> Result<UiReplayTrace, String> {
    replay_session(session).map_err(|err| err.to_string())?;

    let mut belief = session.initial_snapshot.belief.clone();
    let mut steps = Vec::with_capacity(session.steps.len() + 1);
    steps.push(UiReplayStep {
        index: 0,
        board: ui_board_from_visible(&belief.visible),
        move_text: None,
        revealed_card: None,
        backend: session
            .metadata
            .backend
            .map(|backend| format!("{backend:?}")),
        preset: session.metadata.preset_name.clone(),
        planner: None,
    });

    for step in &session.steps {
        let reveal = step.reveal_record.as_ref().map(|record| record.reveal.card);
        let (next_belief, _) =
            apply_observed_belief_move(&belief, step.move_record.macro_move.atomic, reveal)
                .map_err(|err| err.to_string())?;
        belief = next_belief;
        steps.push(UiReplayStep {
            index: step.step_index + 1,
            board: ui_board_from_visible(&belief.visible),
            move_text: Some(format!("{:?}", step.move_record.macro_move.kind)),
            revealed_card: reveal.map(card_text),
            backend: step
                .move_record
                .planner
                .as_ref()
                .map(|planner| format!("{:?}", planner.backend))
                .or_else(|| {
                    session
                        .metadata
                        .backend
                        .map(|backend| format!("{backend:?}"))
                }),
            preset: session.metadata.preset_name.clone(),
            planner: step
                .move_record
                .planner
                .as_ref()
                .map(|planner| UiPlannerSnapshot {
                    best_value: planner.best_value,
                    elapsed_ms: planner.elapsed_ms,
                    deterministic_nodes: planner.deterministic_nodes,
                    root_visits: planner.root_visits,
                    late_exact_triggered: planner.late_exact_triggered,
                    root_parallel_used: false,
                    root_parallel_workers: 1,
                }),
        });
    }

    Ok(UiReplayTrace {
        label: session
            .metadata
            .label
            .clone()
            .unwrap_or_else(|| session.metadata.id.0.to_string()),
        metadata: session_metadata(session),
        steps,
        termination: None,
    })
}

fn session_metadata(session: &SessionRecord) -> UiSessionMetadata {
    UiSessionMetadata {
        session_id: session.metadata.id.0.to_string(),
        label: session.metadata.label.clone(),
        preset_name: session.metadata.preset_name.clone(),
        backend: session
            .metadata
            .backend
            .map(|backend| format!("{backend:?}")),
        schema_version: session.metadata.schema_version.clone(),
        engine_version: session.metadata.engine_version.clone(),
    }
}

fn ui_board_from_visible(visible: &solver_core::VisibleState) -> UiBoardState {
    let foundations = [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades]
        .iter()
        .map(|suit| {
            visible
                .foundations
                .top_rank(*suit)
                .map(|rank| card_text(Card::from_suit_rank(*suit, rank)))
        })
        .collect();
    let tableau = visible
        .columns
        .iter()
        .enumerate()
        .map(|(index, column)| UiTableauColumn {
            index,
            hidden_count: column.hidden_count,
            face_up: column.face_up.iter().copied().map(card_text).collect(),
        })
        .collect();
    UiBoardState {
        foundations,
        tableau,
        stock: UiStockState {
            stock_len: visible.stock.stock_len(),
            waste_len: visible.stock.waste_len(),
            accessible_card: visible.stock.accessible_card().map(card_text),
            pass_index: visible.stock.pass_index,
            draw_count: visible.stock.draw_count,
            ring_len: visible.stock.len(),
        },
        hidden_count: visible.hidden_slot_count(),
        structural_win: visible.is_structural_win(),
    }
}

fn card_text(card: Card) -> String {
    card.to_string()
}

fn parse_json<T: for<'de> Deserialize<'de>>(body: &[u8]) -> Result<T, String> {
    serde_json::from_slice(body).map_err(|err| format!("invalid JSON request: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use solver_core::SessionMetadata;

    fn generation_settings(max_steps: Option<usize>) -> SolverSettingsRequest {
        SolverSettingsRequest {
            preset: "fast_benchmark".to_string(),
            backend: Some("belief_uct".to_string()),
            leaf_eval_mode: Some("heuristic".to_string()),
            vnet_model_path: None,
            max_steps,
            late_exact_enabled: Some(false),
            root_parallel: Some(false),
            root_workers: Some(1),
            worker_sim_budget: Some(1),
            worker_seed_stride: Some(1000003),
        }
    }

    #[test]
    fn ui_board_encoding_is_stable_for_seeded_deal() {
        let deal = ExperimentRunner
            .generate_deal(solver_core::DealSeed(42))
            .unwrap();
        let board = ui_board_from_visible(&deal.full_state.visible);

        assert_eq!(board.tableau.len(), 7);
        assert_eq!(board.foundations.len(), 4);
        assert_eq!(board.stock.draw_count, 3);
    }

    #[test]
    fn configured_preset_applies_root_parallel_override() {
        let preset = configured_preset(&SolverSettingsRequest {
            preset: "fast_benchmark".to_string(),
            backend: Some("belief_uct".to_string()),
            leaf_eval_mode: Some("heuristic".to_string()),
            vnet_model_path: None,
            max_steps: Some(1),
            late_exact_enabled: Some(false),
            root_parallel: Some(true),
            root_workers: Some(2),
            worker_sim_budget: Some(3),
            worker_seed_stride: Some(11),
        })
        .unwrap();

        assert_eq!(preset.autoplay.backend, PlannerBackend::BeliefUct);
        assert!(preset.solver.belief_planner.enable_root_parallel);
        assert_eq!(preset.solver.belief_planner.root_workers, 2);
        assert_eq!(
            preset.solver.belief_planner.worker_simulation_budget,
            Some(3)
        );
        assert_eq!(preset.autoplay.max_steps, 1);
    }

    #[test]
    fn parse_session_returns_replay_trace() {
        let deal = ExperimentRunner
            .generate_deal(solver_core::DealSeed(7))
            .unwrap();
        let session = SessionRecord::from_full_state(
            SessionMetadata::generated(Some("test".to_string())),
            deal.full_state,
        )
        .unwrap();
        let request = SessionParseRequest {
            contents: serde_json::to_string(&session).unwrap(),
            label: None,
        };
        let response = parse_session(&serde_json::to_vec(&request).unwrap()).unwrap();

        assert_eq!(response.replay.steps.len(), 1);
        assert_eq!(response.replay.metadata.label, Some("test".to_string()));
    }

    #[test]
    fn generate_game_returns_valid_session_payload() {
        let request = GenerateGameRequest {
            seed: 19,
            settings: generation_settings(Some(0)),
        };
        let response = generate_game(&serde_json::to_vec(&request).unwrap()).unwrap();

        assert_eq!(response.replay.steps.len(), 1);
        assert_eq!(
            response.generated.as_ref().map(|meta| meta.kind),
            Some(GeneratedGameKind::GeneratedOnly)
        );
        assert!(response.replay.steps[0].board.hidden_count > 0);
        response.session.validate_structure().unwrap();
    }

    #[test]
    fn generate_autoplay_returns_replayable_trace_payload() {
        let request = GenerateGameRequest {
            seed: 23,
            settings: generation_settings(Some(0)),
        };
        let response = generate_autoplay(&serde_json::to_vec(&request).unwrap()).unwrap();

        assert!(!response.replay.steps.is_empty());
        assert_eq!(
            response.generated.as_ref().map(|meta| meta.kind),
            Some(GeneratedGameKind::GeneratedAutoplay)
        );
        replay_session(&response.session).unwrap();
    }

    #[test]
    fn generate_game_is_deterministic_for_same_seed_and_preset() {
        let request = GenerateGameRequest {
            seed: 29,
            settings: generation_settings(Some(0)),
        };
        let body = serde_json::to_vec(&request).unwrap();
        let first = generate_game(&body).unwrap();
        let second = generate_game(&body).unwrap();

        assert_eq!(
            first.session.initial_snapshot.visible,
            second.session.initial_snapshot.visible
        );
        assert_eq!(
            first.session.initial_snapshot.belief.unseen_cards,
            second.session.initial_snapshot.belief.unseen_cards
        );
    }

    #[test]
    fn save_after_generate_round_trips() {
        let request = GenerateGameRequest {
            seed: 31,
            settings: generation_settings(Some(0)),
        };
        let response = generate_game(&serde_json::to_vec(&request).unwrap()).unwrap();
        let path = std::env::temp_dir().join(format!(
            "solitaire-workbench-generated-{}-{}.json",
            std::process::id(),
            response.session.metadata.id.0
        ));
        let save_request = SaveSessionPathRequest {
            path: path.display().to_string(),
            session: response.session.clone(),
        };

        save_session_to_path(&serde_json::to_vec(&save_request).unwrap()).unwrap();
        let load_request = PathRequest {
            path: path.display().to_string(),
        };
        let loaded = load_session_from_path(&serde_json::to_vec(&load_request).unwrap()).unwrap();
        let _ = std::fs::remove_file(path);

        assert_eq!(
            loaded.session.initial_snapshot.visible,
            response.session.initial_snapshot.visible
        );
    }

    #[test]
    fn analyze_current_reports_generated_start_state_shape() {
        let request = GenerateGameRequest {
            seed: 37,
            settings: generation_settings(Some(0)),
        };
        let response = generate_game(&serde_json::to_vec(&request).unwrap()).unwrap();
        let diagnostic_request = DiagnosticRequest {
            session: response.session,
            settings: Some(generation_settings(Some(0))),
        };
        let report = analyze_current(&serde_json::to_vec(&diagnostic_request).unwrap()).unwrap();

        assert!(report.replay_matched);
        assert!(report.consistency_held);
        assert_eq!(report.steps.len(), 0);
        assert_eq!(
            report.final_termination_reason,
            "StoppedWithLegalMovesRemaining"
        );
        assert!(report.terminal_audit.legal_moves_remaining > 0);
    }

    #[test]
    fn analyze_generated_autoplay_session_returns_valid_report() {
        let request = GenerateGameRequest {
            seed: 41,
            settings: generation_settings(Some(1)),
        };
        let response = generate_autoplay(&serde_json::to_vec(&request).unwrap()).unwrap();
        let step_count = response.session.steps.len();
        let diagnostic_request = DiagnosticRequest {
            session: response.session,
            settings: Some(generation_settings(Some(1))),
        };
        let report = analyze_session(&serde_json::to_vec(&diagnostic_request).unwrap()).unwrap();

        assert!(report.replay_matched);
        assert_eq!(report.steps.len(), step_count);
        assert!(!report.overall_status.is_empty());
        for step in &report.steps {
            assert!(!step.chosen_move.is_empty());
            assert!(step.legal_move_count > 0);
        }
    }
}
