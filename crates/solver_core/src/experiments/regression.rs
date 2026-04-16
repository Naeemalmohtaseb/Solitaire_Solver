//! Curated regression packs for solver behavior changes.
//!
//! Regression packs are durable, versioned JSON fixtures. They snapshot
//! interesting deterministic, belief-root, autoplay, oracle, or replay cases and
//! explicit expectations so later solver versions can detect behavior changes.

use std::{collections::BTreeMap, fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    core::{BeliefState, FullState},
    deterministic_solver::{
        DeterministicSearchConfig, DeterministicSearchStats, DeterministicSolver, SolveOutcome,
    },
    error::{SolverError, SolverResult},
    moves::MacroMoveKind,
    planner::recommend_move_belief_uct,
    session::{replay_session, SessionRecord},
    types::DealSeed,
};

use super::{
    csv_table, deterministic_search_config_from_solver, oracle_cases_from_seeded_suite,
    play_game_with_planner, to_pretty_json, AutoplayResult, AutoplayTermination, BenchmarkSuite,
    ExperimentPreset, OracleCaseResult, OracleEvaluationMode, OracleReferenceResult,
};

/// JSON schema version for regression packs.
pub const REGRESSION_PACK_SCHEMA_VERSION: &str = "solitaire-regression-pack-v1";

/// Metadata for a curated regression pack.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegressionPackMetadata {
    /// Pack schema version.
    pub schema_version: String,
    /// Stable pack name.
    pub name: String,
    /// Optional human-readable description.
    pub description: Option<String>,
    /// Solver crate version that wrote this pack.
    pub engine_version: String,
}

impl RegressionPackMetadata {
    /// Creates metadata for a new pack.
    pub fn new(name: impl Into<String>, description: Option<String>) -> Self {
        Self {
            schema_version: REGRESSION_PACK_SCHEMA_VERSION.to_string(),
            name: name.into(),
            description,
            engine_version: crate::VERSION.to_string(),
        }
    }
}

/// Versioned collection of regression cases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionPack {
    /// Pack metadata.
    pub metadata: RegressionPackMetadata,
    /// Cases in deterministic run order.
    pub cases: Vec<RegressionCase>,
}

impl RegressionPack {
    /// Creates a regression pack.
    pub fn new(metadata: RegressionPackMetadata, cases: Vec<RegressionCase>) -> Self {
        Self { metadata, cases }
    }

    /// Serializes the pack as pretty JSON.
    pub fn to_json(&self) -> SolverResult<String> {
        to_pretty_json(self)
    }

    /// Returns compact summary metadata.
    pub fn summary(&self) -> RegressionPackSummary {
        let mut tag_counts = BTreeMap::<String, usize>::new();
        let mut kind_counts = BTreeMap::<String, usize>::new();
        for case in &self.cases {
            *kind_counts
                .entry(format!("{:?}", case.kind.case_kind()))
                .or_insert(0) += 1;
            for tag in &case.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        RegressionPackSummary {
            name: self.metadata.name.clone(),
            schema_version: self.metadata.schema_version.clone(),
            engine_version: self.metadata.engine_version.clone(),
            case_count: self.cases.len(),
            tag_counts,
            kind_counts,
        }
    }
}

/// Compact regression-pack summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegressionPackSummary {
    /// Pack name.
    pub name: String,
    /// Schema version.
    pub schema_version: String,
    /// Writer engine version.
    pub engine_version: String,
    /// Number of cases.
    pub case_count: usize,
    /// Number of cases per tag.
    pub tag_counts: BTreeMap<String, usize>,
    /// Number of cases per kind.
    pub kind_counts: BTreeMap<String, usize>,
}

/// Durable source/provenance metadata for one regression case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegressionCaseProvenance {
    /// Human-readable source label.
    pub source_label: String,
    /// Optional preset name.
    pub preset_name: Option<String>,
    /// Optional deal seed.
    pub seed: Option<DealSeed>,
    /// Optional source case id, session id, or oracle case id.
    pub source_id: Option<String>,
}

impl RegressionCaseProvenance {
    /// Creates provenance with a source label.
    pub fn new(source_label: impl Into<String>) -> Self {
        Self {
            source_label: source_label.into(),
            preset_name: None,
            seed: None,
            source_id: None,
        }
    }
}

/// Regression case kind discriminator.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegressionCaseKind {
    /// Fully known deterministic open-card state.
    DeterministicOpenCard,
    /// Hidden-information belief root.
    HiddenInformationRoot,
    /// Full-game autoplay start state.
    Autoplay,
    /// Offline oracle comparison bundle.
    OracleComparison,
    /// Persisted session replay.
    SessionReplay,
}

/// Case payload variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionCaseData {
    /// Fully known deterministic open-card state.
    DeterministicOpenCard {
        /// Full deterministic state.
        full_state: FullState,
    },
    /// Hidden-information belief/root state.
    HiddenInformationRoot {
        /// Belief state.
        belief: BeliefState,
    },
    /// Full-game autoplay state.
    Autoplay {
        /// Initial true full state.
        full_state: FullState,
    },
    /// Local/reference oracle rows to compare.
    OracleComparison {
        /// Local deterministic rows.
        local: Vec<OracleCaseResult>,
        /// External/reference rows.
        reference: Vec<OracleReferenceResult>,
    },
    /// Persisted replayable session.
    SessionReplay {
        /// Session record.
        session: SessionRecord,
    },
}

impl RegressionCaseData {
    const fn case_kind(&self) -> RegressionCaseKind {
        match self {
            Self::DeterministicOpenCard { .. } => RegressionCaseKind::DeterministicOpenCard,
            Self::HiddenInformationRoot { .. } => RegressionCaseKind::HiddenInformationRoot,
            Self::Autoplay { .. } => RegressionCaseKind::Autoplay,
            Self::OracleComparison { .. } => RegressionCaseKind::OracleComparison,
            Self::SessionReplay { .. } => RegressionCaseKind::SessionReplay,
        }
    }
}

/// Explicit expectation checked by the regression runner.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionExpectation {
    /// Expected deterministic outcome.
    DeterministicOutcome {
        /// Expected outcome.
        outcome: SolveOutcome,
    },
    /// Expected deterministic best move.
    DeterministicBestMove {
        /// Expected best move kind.
        best_move: Option<MacroMoveKind>,
    },
    /// Expected planner/root selected move.
    ChosenMove {
        /// Preset/backend expectation applies to.
        preset_name: Option<String>,
        /// Expected chosen move.
        best_move: Option<MacroMoveKind>,
    },
    /// Expected autoplay win bit.
    AutoplayWon {
        /// Expected win bit.
        won: bool,
    },
    /// Expected autoplay termination.
    AutoplayTermination {
        /// Expected termination reason.
        termination: AutoplayTermination,
    },
    /// Expected replay consistency.
    ReplayConsistency {
        /// Expected replay match flag.
        matched: bool,
    },
    /// Expected oracle mismatch count.
    OracleMismatchCount {
        /// Expected mismatch count.
        mismatches: usize,
    },
    /// Expected approximate value with tolerance.
    ApproxValue {
        /// Expected value.
        value: f64,
        /// Absolute tolerance.
        tolerance: f64,
    },
}

/// One curated regression case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionCase {
    /// Stable case id.
    pub case_id: String,
    /// Case payload.
    pub kind: RegressionCaseData,
    /// Source/provenance metadata.
    pub provenance: RegressionCaseProvenance,
    /// Lightweight tags such as `reveal-heavy` or `oracle-mismatch`.
    pub tags: Vec<String>,
    /// Explicit expectations checked during runs.
    pub expectations: Vec<RegressionExpectation>,
}

impl RegressionCase {
    /// Creates a regression case.
    pub fn new(
        case_id: impl Into<String>,
        kind: RegressionCaseData,
        provenance: RegressionCaseProvenance,
        tags: Vec<String>,
        expectations: Vec<RegressionExpectation>,
    ) -> Self {
        Self {
            case_id: case_id.into(),
            kind,
            provenance,
            tags: normalize_tags(tags),
            expectations,
        }
    }
}

/// Runner configuration for one pack execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionRunConfig {
    /// Preset used for planner/autoplay checks.
    pub preset: ExperimentPreset,
    /// Deterministic local evaluation mode.
    pub deterministic_mode: OracleEvaluationMode,
    /// Optional deterministic search override.
    pub deterministic_override: Option<DeterministicSearchConfig>,
}

impl RegressionRunConfig {
    /// Builds a run config from a preset.
    pub fn from_preset(preset: ExperimentPreset) -> Self {
        Self {
            preset,
            deterministic_mode: OracleEvaluationMode::Fast,
            deterministic_override: None,
        }
    }

    fn deterministic_config(&self) -> DeterministicSearchConfig {
        self.deterministic_override
            .unwrap_or_else(|| deterministic_search_config_from_solver(&self.preset.solver))
    }
}

/// Mismatch category for regression results.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RegressionMismatchKind {
    /// Case has no expectations.
    MissingExpectation,
    /// Deterministic outcome changed.
    OutcomeMismatch,
    /// Deterministic best move changed.
    BestMoveMismatch,
    /// Planner/root chosen move changed.
    ChosenMoveMismatch,
    /// Autoplay win or termination changed.
    AutoplayMismatch,
    /// Replay consistency changed.
    ReplayMismatch,
    /// Oracle comparison mismatch count changed.
    OracleMismatch,
    /// Approximate value moved outside tolerance.
    ValueMismatch,
    /// Expectation does not apply to this case kind/config.
    UnsupportedCaseConfig,
}

/// One mismatch diagnostic.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionMismatch {
    /// Mismatch category.
    pub kind: RegressionMismatchKind,
    /// Human-readable detail.
    pub message: String,
}

/// Observed behavior for one regression case.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RegressionObserved {
    /// Observed deterministic outcome.
    pub deterministic_outcome: Option<SolveOutcome>,
    /// Observed deterministic best move.
    pub deterministic_best_move: Option<MacroMoveKind>,
    /// Observed planner chosen move.
    pub chosen_move: Option<MacroMoveKind>,
    /// Observed autoplay win bit.
    pub autoplay_won: Option<bool>,
    /// Observed autoplay termination.
    pub autoplay_termination: Option<AutoplayTermination>,
    /// Observed replay match flag.
    pub replay_matched: Option<bool>,
    /// Observed oracle comparison mismatch count.
    pub oracle_mismatches: Option<usize>,
    /// Observed scalar value.
    pub value: Option<f64>,
    /// Deterministic search stats, when applicable.
    pub deterministic_stats: Option<DeterministicSearchStats>,
}

/// Result for one case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionCaseResult {
    /// Case id.
    pub case_id: String,
    /// Case kind.
    pub case_kind: RegressionCaseKind,
    /// Case tags.
    pub tags: Vec<String>,
    /// Observed behavior.
    pub observed: RegressionObserved,
    /// Mismatch diagnostics.
    pub mismatches: Vec<RegressionMismatch>,
    /// True if no mismatches were found.
    pub passed: bool,
}

/// Count of one regression mismatch category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegressionMismatchCount {
    /// Mismatch kind.
    pub kind: RegressionMismatchKind,
    /// Number of cases with this kind.
    pub count: usize,
}

/// Result of running a regression pack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegressionRunResult {
    /// Pack name.
    pub pack_name: String,
    /// Preset used for this run.
    pub preset_name: String,
    /// Total cases.
    pub total_cases: usize,
    /// Passed cases.
    pub passed: usize,
    /// Failed cases.
    pub failed: usize,
    /// Per-kind mismatch counts.
    pub mismatch_counts: Vec<RegressionMismatchCount>,
    /// Per-case results.
    pub case_results: Vec<RegressionCaseResult>,
}

impl RegressionRunResult {
    /// Exports a deterministic JSON summary.
    pub fn to_json_summary(&self) -> SolverResult<String> {
        to_pretty_json(self)
    }

    /// Exports a deterministic CSV summary with one row per case.
    pub fn to_csv_summary(&self) -> String {
        let rows = self
            .case_results
            .iter()
            .map(|case| {
                vec![
                    case.case_id.clone(),
                    format!("{:?}", case.case_kind),
                    case.passed.to_string(),
                    case.tags.join(";"),
                    case.mismatches
                        .iter()
                        .map(|mismatch| format!("{:?}", mismatch.kind))
                        .collect::<Vec<_>>()
                        .join(";"),
                    case.observed
                        .deterministic_outcome
                        .map(|outcome| format!("{outcome:?}"))
                        .unwrap_or_default(),
                    case.observed
                        .deterministic_best_move
                        .map(|best| format!("{best:?}"))
                        .unwrap_or_default(),
                    case.observed
                        .chosen_move
                        .map(|best| format!("{best:?}"))
                        .unwrap_or_default(),
                    case.observed
                        .autoplay_won
                        .map(|won| won.to_string())
                        .unwrap_or_default(),
                    case.observed
                        .replay_matched
                        .map(|matched| matched.to_string())
                        .unwrap_or_default(),
                    case.observed
                        .oracle_mismatches
                        .map(|count| count.to_string())
                        .unwrap_or_default(),
                    case.observed
                        .value
                        .map(|value| value.to_string())
                        .unwrap_or_default(),
                ]
            })
            .collect::<Vec<_>>();
        csv_table(
            &[
                "case_id",
                "case_kind",
                "passed",
                "tags",
                "mismatch_kinds",
                "deterministic_outcome",
                "deterministic_best_move",
                "chosen_move",
                "autoplay_won",
                "replay_matched",
                "oracle_mismatches",
                "value",
            ],
            &rows,
        )
    }
}

/// Creates a deterministic-open-card regression pack from a seeded suite and
/// snapshots the current solver behavior as explicit expectations.
pub fn regression_pack_from_benchmark_suite(
    suite: &BenchmarkSuite,
    config: &RegressionRunConfig,
    name: impl Into<String>,
    tags: Vec<String>,
) -> SolverResult<RegressionPack> {
    let oracle_cases = oracle_cases_from_seeded_suite(suite, Some(&config.preset))?;
    let mut cases = Vec::with_capacity(oracle_cases.cases.len());
    for oracle_case in oracle_cases.cases {
        let mut provenance = RegressionCaseProvenance::new("benchmark_suite");
        provenance.preset_name = Some(config.preset.name.clone());
        provenance.seed = oracle_case.provenance.seed;
        provenance.source_id = Some(oracle_case.case_id.clone());
        let mut case = RegressionCase::new(
            oracle_case.case_id,
            RegressionCaseData::DeterministicOpenCard {
                full_state: oracle_case.full_state,
            },
            provenance,
            tags.clone(),
            Vec::new(),
        );
        let observed = observe_case(&case, config)?;
        case.expectations = expectations_from_observed(&observed, &case.kind, Some(&config.preset));
        cases.push(case);
    }
    Ok(RegressionPack::new(
        RegressionPackMetadata::new(
            name,
            Some("created from seeded benchmark suite".to_string()),
        ),
        cases,
    ))
}

/// Creates a deterministic open-card regression case from an explicit full state.
pub fn regression_case_from_full_state(
    case_id: impl Into<String>,
    full_state: FullState,
    provenance: RegressionCaseProvenance,
    tags: Vec<String>,
    expectations: Vec<RegressionExpectation>,
) -> RegressionCase {
    RegressionCase::new(
        case_id,
        RegressionCaseData::DeterministicOpenCard { full_state },
        provenance,
        tags,
        expectations,
    )
}

/// Creates a hidden-information root regression case from an explicit belief state.
pub fn regression_case_from_belief_state(
    case_id: impl Into<String>,
    belief: BeliefState,
    provenance: RegressionCaseProvenance,
    tags: Vec<String>,
    expectations: Vec<RegressionExpectation>,
) -> RegressionCase {
    RegressionCase::new(
        case_id,
        RegressionCaseData::HiddenInformationRoot { belief },
        provenance,
        tags,
        expectations,
    )
}

/// Creates a full-game autoplay regression case from an explicit true full state.
pub fn regression_case_from_autoplay_state(
    case_id: impl Into<String>,
    full_state: FullState,
    provenance: RegressionCaseProvenance,
    tags: Vec<String>,
    expectations: Vec<RegressionExpectation>,
) -> RegressionCase {
    RegressionCase::new(
        case_id,
        RegressionCaseData::Autoplay { full_state },
        provenance,
        tags,
        expectations,
    )
}

/// Creates an autoplay regression pack from an already-run autoplay result and
/// the corresponding initial full state.
pub fn regression_pack_from_autoplay_result(
    initial_full_state: FullState,
    result: &AutoplayResult,
    name: impl Into<String>,
    tags: Vec<String>,
) -> RegressionPack {
    let case = regression_case_from_autoplay_state(
        "autoplay:0",
        initial_full_state,
        RegressionCaseProvenance::new("autoplay_trace"),
        tags,
        vec![
            RegressionExpectation::AutoplayWon { won: result.won },
            RegressionExpectation::AutoplayTermination {
                termination: result.termination,
            },
        ],
    );
    RegressionPack::new(
        RegressionPackMetadata::new(name, Some("created from autoplay trace".to_string())),
        vec![case],
    )
}

/// Creates a replay regression pack from a persisted session.
pub fn regression_pack_from_session(
    session: SessionRecord,
    name: impl Into<String>,
    tags: Vec<String>,
) -> RegressionPack {
    let mut provenance = RegressionCaseProvenance::new("session");
    provenance.preset_name = session.metadata.preset_name.clone();
    provenance.source_id = Some(session.metadata.id.0.to_string());
    let case = RegressionCase::new(
        format!("session:{}", session.metadata.id.0),
        RegressionCaseData::SessionReplay { session },
        provenance,
        tags,
        vec![RegressionExpectation::ReplayConsistency { matched: true }],
    );
    RegressionPack::new(
        RegressionPackMetadata::new(name, Some("created from session replay".to_string())),
        vec![case],
    )
}

/// Creates one oracle-comparison regression case with the current mismatch count
/// as the expectation.
pub fn regression_case_from_oracle_comparison(
    case_id: impl Into<String>,
    local: Vec<OracleCaseResult>,
    reference: Vec<OracleReferenceResult>,
    tags: Vec<String>,
) -> RegressionCase {
    let summary = super::compare_oracle_results(&local, &reference);
    RegressionCase::new(
        case_id,
        RegressionCaseData::OracleComparison { local, reference },
        RegressionCaseProvenance::new("oracle_comparison"),
        tags,
        vec![RegressionExpectation::OracleMismatchCount {
            mismatches: summary.mismatches,
        }],
    )
}

/// Runs a regression pack against the supplied current solver configuration.
pub fn run_regression_pack(
    pack: &RegressionPack,
    config: &RegressionRunConfig,
) -> SolverResult<RegressionRunResult> {
    let mut case_results = Vec::with_capacity(pack.cases.len());
    for case in &pack.cases {
        let observed = observe_case(case, config)?;
        let mismatches = compare_expectations(case, &observed, Some(&config.preset));
        case_results.push(RegressionCaseResult {
            case_id: case.case_id.clone(),
            case_kind: case.kind.case_kind(),
            tags: case.tags.clone(),
            passed: mismatches.is_empty(),
            observed,
            mismatches,
        });
    }
    let total_cases = case_results.len();
    let passed = case_results.iter().filter(|case| case.passed).count();
    let failed = total_cases.saturating_sub(passed);
    let mismatch_counts = summarize_regression_mismatches(&case_results);
    Ok(RegressionRunResult {
        pack_name: pack.metadata.name.clone(),
        preset_name: config.preset.name.clone(),
        total_cases,
        passed,
        failed,
        mismatch_counts,
        case_results,
    })
}

/// Saves a regression pack as JSON.
pub fn save_regression_pack(path: impl AsRef<Path>, pack: &RegressionPack) -> SolverResult<()> {
    fs::write(path, pack.to_json()?)?;
    Ok(())
}

/// Loads a regression pack from JSON.
pub fn load_regression_pack(path: impl AsRef<Path>) -> SolverResult<RegressionPack> {
    let contents = fs::read_to_string(path)?;
    let pack: RegressionPack = serde_json::from_str(&contents)
        .map_err(|error| SolverError::Serialization(error.to_string()))?;
    if pack.metadata.schema_version != REGRESSION_PACK_SCHEMA_VERSION {
        return Err(SolverError::InvalidState(format!(
            "unsupported regression pack schema {}",
            pack.metadata.schema_version
        )));
    }
    Ok(pack)
}

fn observe_case(
    case: &RegressionCase,
    config: &RegressionRunConfig,
) -> SolverResult<RegressionObserved> {
    match &case.kind {
        RegressionCaseData::DeterministicOpenCard { full_state } => {
            observe_deterministic_case(full_state, config)
        }
        RegressionCaseData::HiddenInformationRoot { belief } => {
            let recommendation = recommend_move_belief_uct(
                belief,
                &config.preset.solver,
                &config.preset.solver.belief_planner,
            )?;
            Ok(RegressionObserved {
                chosen_move: recommendation.best_move.map(|best| best.kind),
                value: Some(recommendation.best_value),
                ..RegressionObserved::default()
            })
        }
        RegressionCaseData::Autoplay { full_state } => {
            let result =
                play_game_with_planner(full_state, &config.preset.solver, &config.preset.autoplay)?;
            Ok(RegressionObserved {
                autoplay_won: Some(result.won),
                autoplay_termination: Some(result.termination),
                value: Some(if result.won { 1.0 } else { 0.0 }),
                ..RegressionObserved::default()
            })
        }
        RegressionCaseData::OracleComparison { local, reference } => {
            let summary = super::compare_oracle_results(local, reference);
            Ok(RegressionObserved {
                oracle_mismatches: Some(summary.mismatches),
                value: Some(summary.matches as f64),
                ..RegressionObserved::default()
            })
        }
        RegressionCaseData::SessionReplay { session } => {
            let replay = replay_session(session)?;
            Ok(RegressionObserved {
                replay_matched: Some(replay.matched),
                value: Some(if replay.matched { 1.0 } else { 0.0 }),
                ..RegressionObserved::default()
            })
        }
    }
}

fn observe_deterministic_case(
    full_state: &FullState,
    config: &RegressionRunConfig,
) -> SolverResult<RegressionObserved> {
    let solver = DeterministicSolver::new(config.deterministic_config());
    let (outcome, best_move, value, stats) = match config.deterministic_mode {
        OracleEvaluationMode::Exact => {
            let result = solver.solve_exact(full_state)?;
            (
                result.outcome,
                result.best_move.map(|best| best.kind),
                f64::from(result.value),
                result.stats,
            )
        }
        OracleEvaluationMode::Bounded => {
            let result = solver.solve_bounded(full_state)?;
            (
                result.outcome,
                result.best_move.map(|best| best.kind),
                f64::from(result.estimated_value),
                result.stats,
            )
        }
        OracleEvaluationMode::Fast => {
            let result = solver.evaluate_fast(full_state)?;
            (
                if full_state.visible.is_structural_win() {
                    SolveOutcome::ProvenWin
                } else {
                    SolveOutcome::Unknown
                },
                result.best_move.map(|best| best.kind),
                f64::from(result.value),
                result.stats,
            )
        }
    };
    Ok(RegressionObserved {
        deterministic_outcome: Some(outcome),
        deterministic_best_move: best_move,
        value: Some(value),
        deterministic_stats: Some(stats),
        ..RegressionObserved::default()
    })
}

fn expectations_from_observed(
    observed: &RegressionObserved,
    case_data: &RegressionCaseData,
    preset: Option<&ExperimentPreset>,
) -> Vec<RegressionExpectation> {
    let mut expectations = Vec::new();
    match case_data {
        RegressionCaseData::DeterministicOpenCard { .. } => {
            if let Some(outcome) = observed.deterministic_outcome {
                expectations.push(RegressionExpectation::DeterministicOutcome { outcome });
            }
            expectations.push(RegressionExpectation::DeterministicBestMove {
                best_move: observed.deterministic_best_move,
            });
            if let Some(value) = observed.value {
                expectations.push(RegressionExpectation::ApproxValue {
                    value,
                    tolerance: 0.000_001,
                });
            }
        }
        RegressionCaseData::HiddenInformationRoot { .. } => {
            expectations.push(RegressionExpectation::ChosenMove {
                preset_name: preset.map(|preset| preset.name.clone()),
                best_move: observed.chosen_move,
            });
        }
        RegressionCaseData::Autoplay { .. } => {
            if let Some(won) = observed.autoplay_won {
                expectations.push(RegressionExpectation::AutoplayWon { won });
            }
            if let Some(termination) = observed.autoplay_termination {
                expectations.push(RegressionExpectation::AutoplayTermination { termination });
            }
        }
        RegressionCaseData::OracleComparison { .. } => {
            if let Some(mismatches) = observed.oracle_mismatches {
                expectations.push(RegressionExpectation::OracleMismatchCount { mismatches });
            }
        }
        RegressionCaseData::SessionReplay { .. } => {
            if let Some(matched) = observed.replay_matched {
                expectations.push(RegressionExpectation::ReplayConsistency { matched });
            }
        }
    }
    expectations
}

fn compare_expectations(
    case: &RegressionCase,
    observed: &RegressionObserved,
    preset: Option<&ExperimentPreset>,
) -> Vec<RegressionMismatch> {
    if case.expectations.is_empty() {
        return vec![RegressionMismatch {
            kind: RegressionMismatchKind::MissingExpectation,
            message: "regression case has no expectations".to_string(),
        }];
    }

    let mut mismatches = Vec::new();
    for expectation in &case.expectations {
        match expectation {
            RegressionExpectation::DeterministicOutcome { outcome } => {
                compare_option(
                    observed.deterministic_outcome,
                    Some(*outcome),
                    RegressionMismatchKind::OutcomeMismatch,
                    "deterministic outcome",
                    &mut mismatches,
                );
            }
            RegressionExpectation::DeterministicBestMove { best_move } => {
                compare_option(
                    observed.deterministic_best_move,
                    *best_move,
                    RegressionMismatchKind::BestMoveMismatch,
                    "deterministic best move",
                    &mut mismatches,
                );
            }
            RegressionExpectation::ChosenMove {
                preset_name,
                best_move,
            } => {
                if preset_name
                    .as_ref()
                    .zip(preset)
                    .is_some_and(|(expected, actual)| expected != &actual.name)
                {
                    mismatches.push(RegressionMismatch {
                        kind: RegressionMismatchKind::UnsupportedCaseConfig,
                        message: format!(
                            "expectation is for preset {:?}, run used {:?}",
                            preset_name,
                            preset.map(|preset| preset.name.as_str())
                        ),
                    });
                } else {
                    compare_option(
                        observed.chosen_move,
                        *best_move,
                        RegressionMismatchKind::ChosenMoveMismatch,
                        "chosen move",
                        &mut mismatches,
                    );
                }
            }
            RegressionExpectation::AutoplayWon { won } => compare_option(
                observed.autoplay_won,
                Some(*won),
                RegressionMismatchKind::AutoplayMismatch,
                "autoplay win",
                &mut mismatches,
            ),
            RegressionExpectation::AutoplayTermination { termination } => compare_option(
                observed.autoplay_termination,
                Some(*termination),
                RegressionMismatchKind::AutoplayMismatch,
                "autoplay termination",
                &mut mismatches,
            ),
            RegressionExpectation::ReplayConsistency { matched } => compare_option(
                observed.replay_matched,
                Some(*matched),
                RegressionMismatchKind::ReplayMismatch,
                "replay consistency",
                &mut mismatches,
            ),
            RegressionExpectation::OracleMismatchCount {
                mismatches: expected,
            } => compare_option(
                observed.oracle_mismatches,
                Some(*expected),
                RegressionMismatchKind::OracleMismatch,
                "oracle mismatch count",
                &mut mismatches,
            ),
            RegressionExpectation::ApproxValue { value, tolerance } => {
                let Some(observed_value) = observed.value else {
                    mismatches.push(RegressionMismatch {
                        kind: RegressionMismatchKind::UnsupportedCaseConfig,
                        message: "no observed value for approximate value expectation".to_string(),
                    });
                    continue;
                };
                if (observed_value - value).abs() > *tolerance {
                    mismatches.push(RegressionMismatch {
                        kind: RegressionMismatchKind::ValueMismatch,
                        message: format!(
                            "value expected {value} +/- {tolerance}, observed {observed_value}"
                        ),
                    });
                }
            }
        }
    }
    mismatches
}

fn compare_option<T>(
    observed: Option<T>,
    expected: Option<T>,
    kind: RegressionMismatchKind,
    label: &str,
    mismatches: &mut Vec<RegressionMismatch>,
) where
    T: std::fmt::Debug + PartialEq,
{
    if observed != expected {
        mismatches.push(RegressionMismatch {
            kind,
            message: format!("{label} expected {expected:?}, observed {observed:?}"),
        });
    }
}

fn summarize_regression_mismatches(
    results: &[RegressionCaseResult],
) -> Vec<RegressionMismatchCount> {
    let mut counts = BTreeMap::<RegressionMismatchKind, usize>::new();
    for kind in results
        .iter()
        .flat_map(|result| result.mismatches.iter().map(|mismatch| mismatch.kind))
    {
        *counts.entry(kind).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(kind, count)| RegressionMismatchCount { kind, count })
        .collect()
}

fn normalize_tags(mut tags: Vec<String>) -> Vec<String> {
    tags.sort();
    tags.dedup();
    tags
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cards::{Rank, Suit},
        core::{FoundationState, HiddenAssignments, VisibleState},
        session::{SessionMetadata, SessionRecord},
        types::SessionId,
    };

    fn fast_config() -> RegressionRunConfig {
        let mut preset = super::super::fast_benchmark();
        preset.autoplay.max_steps = 0;
        RegressionRunConfig {
            preset,
            deterministic_mode: OracleEvaluationMode::Fast,
            deterministic_override: Some(DeterministicSearchConfig {
                budget: crate::deterministic_solver::SolveBudget {
                    node_budget: Some(16),
                    depth_budget: Some(1),
                    wall_clock_limit_ms: None,
                },
                ..DeterministicSearchConfig::default()
            }),
        }
    }

    fn won_state() -> FullState {
        let mut visible = VisibleState::default();
        let mut foundations = FoundationState::default();
        for suit in [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
            foundations.set_top_rank(suit, Some(Rank::King));
        }
        visible.foundations = foundations;
        FullState::new(visible, HiddenAssignments::empty())
    }

    fn session_record() -> SessionRecord {
        SessionRecord::from_full_state(
            SessionMetadata::new(SessionId(99), Some("regression".to_string())),
            won_state(),
        )
        .unwrap()
    }

    #[test]
    fn regression_pack_serializes_round_trip() {
        let case = RegressionCase::new(
            "won",
            RegressionCaseData::DeterministicOpenCard {
                full_state: won_state(),
            },
            RegressionCaseProvenance::new("unit"),
            vec!["late-exact".to_string(), "late-exact".to_string()],
            vec![RegressionExpectation::DeterministicOutcome {
                outcome: SolveOutcome::ProvenWin,
            }],
        );
        let pack = RegressionPack::new(RegressionPackMetadata::new("unit", None), vec![case]);

        let json = pack.to_json().unwrap();
        let decoded: RegressionPack = serde_json::from_str(&json).unwrap();

        assert_eq!(decoded, pack);
        assert_eq!(decoded.summary().tag_counts.get("late-exact"), Some(&1));
    }

    #[test]
    fn regression_run_is_reproducible() {
        let suite = BenchmarkSuite::from_base_seed("regression-suite", 31, 1);
        let config = fast_config();
        let pack = regression_pack_from_benchmark_suite(
            &suite,
            &config,
            "suite-pack",
            vec!["hard".into()],
        )
        .unwrap();

        let first = run_regression_pack(&pack, &config).unwrap();
        let second = run_regression_pack(&pack, &config).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.total_cases, 1);
        assert_eq!(first.failed, 0);
    }

    #[test]
    fn mismatch_classification_detects_outcome_change() {
        let case = RegressionCase::new(
            "won",
            RegressionCaseData::DeterministicOpenCard {
                full_state: won_state(),
            },
            RegressionCaseProvenance::new("unit"),
            Vec::new(),
            vec![RegressionExpectation::DeterministicOutcome {
                outcome: SolveOutcome::ProvenLoss,
            }],
        );
        let pack = RegressionPack::new(RegressionPackMetadata::new("unit", None), vec![case]);

        let result = run_regression_pack(&pack, &fast_config()).unwrap();

        assert_eq!(result.failed, 1);
        assert_eq!(
            result.case_results[0].mismatches[0].kind,
            RegressionMismatchKind::OutcomeMismatch
        );
    }

    #[test]
    fn session_creation_helper_produces_valid_replay_case() {
        let pack =
            regression_pack_from_session(session_record(), "session-pack", vec!["replay".into()]);

        let result = run_regression_pack(&pack, &fast_config()).unwrap();

        assert_eq!(pack.cases.len(), 1);
        assert_eq!(result.failed, 0);
        assert_eq!(pack.cases[0].tags, vec!["replay"]);
    }

    #[test]
    fn explicit_creation_helpers_produce_valid_case_kinds() {
        let full = won_state();
        let deterministic = regression_case_from_full_state(
            "full",
            full.clone(),
            RegressionCaseProvenance::new("manual"),
            vec!["foundation-trap".into()],
            Vec::new(),
        );
        assert_eq!(
            deterministic.kind.case_kind(),
            RegressionCaseKind::DeterministicOpenCard
        );

        let belief = crate::belief::belief_from_full_state(&full).unwrap();
        let belief_case = regression_case_from_belief_state(
            "belief",
            belief.clone(),
            RegressionCaseProvenance::new("manual"),
            vec!["reveal-heavy".into()],
            Vec::new(),
        );
        assert_eq!(
            belief_case.kind.case_kind(),
            RegressionCaseKind::HiddenInformationRoot
        );

        let autoplay_result = AutoplayResult {
            won: true,
            termination: AutoplayTermination::Win,
            trace: crate::experiments::AutoplayTrace::default(),
            final_belief: belief,
            final_full_state: full.clone(),
            total_planner_time_ms: 0,
            deterministic_nodes: 0,
            root_visits: 0,
            root_parallel_steps: 0,
            root_parallel_worker_count: 0,
            root_parallel_simulations: 0,
            late_exact_triggers: 0,
            vnet_inferences: 0,
            vnet_fallbacks: 0,
            vnet_inference_elapsed_us: 0,
        };
        let pack = regression_pack_from_autoplay_result(
            full,
            &autoplay_result,
            "autoplay-pack",
            vec!["late-exact".into()],
        );

        assert_eq!(pack.cases.len(), 1);
        assert_eq!(pack.cases[0].kind.case_kind(), RegressionCaseKind::Autoplay);
        assert!(pack.cases[0]
            .expectations
            .contains(&RegressionExpectation::AutoplayWon { won: true }));
    }

    #[test]
    fn oracle_comparison_case_expectation_is_checked() {
        let local = vec![OracleCaseResult {
            case_id: "x".to_string(),
            provenance: crate::experiments::OracleCaseProvenance::new("unit"),
            outcome: SolveOutcome::ProvenWin,
            best_move: None,
            value: 1.0,
            stats: DeterministicSearchStats::default(),
        }];
        let reference = vec![OracleReferenceResult {
            case_id: "x".to_string(),
            outcome: SolveOutcome::ProvenLoss,
            best_move: None,
            value: Some(0.0),
            source_label: None,
            notes: None,
        }];
        let case = regression_case_from_oracle_comparison(
            "oracle-case",
            local,
            reference,
            vec!["oracle-mismatch".into()],
        );
        let pack = RegressionPack::new(RegressionPackMetadata::new("oracle", None), vec![case]);

        let result = run_regression_pack(&pack, &fast_config()).unwrap();

        assert_eq!(result.failed, 0);
        assert_eq!(result.case_results[0].observed.oracle_mismatches, Some(1));
    }
}
