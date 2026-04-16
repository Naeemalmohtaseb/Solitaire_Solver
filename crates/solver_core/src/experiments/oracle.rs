//! Offline deterministic-solver oracle case export and comparison.
//!
//! This module deliberately avoids invoking or depending on any external
//! solitaire solver. It provides deterministic case/result formats so an
//! external reference can be run out-of-process and compared later by case id.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::{
    core::FullState,
    deterministic_solver::{
        DeterministicSearchConfig, DeterministicSearchStats, DeterministicSolver, SolveOutcome,
    },
    error::{SolverError, SolverResult},
    moves::{apply_atomic_move_full_state, MacroMoveKind},
    types::DealSeed,
};

use super::{
    csv_table, to_pretty_json, AutoplayTrace, BenchmarkSuite, ExperimentPreset, ExperimentRunner,
};

/// JSON schema version for oracle case packs.
pub const ORACLE_CASE_SCHEMA_VERSION: &str = "solitaire-oracle-cases-v1";

/// JSON schema version for local/reference result packs.
pub const ORACLE_RESULT_SCHEMA_VERSION: &str = "solitaire-oracle-results-v1";

/// Metadata describing where an oracle case came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleCaseProvenance {
    /// Human-readable source label such as `benchmark` or `autoplay`.
    pub source_label: String,
    /// Optional preset/config name.
    pub preset_name: Option<String>,
    /// Optional deal seed.
    pub seed: Option<DealSeed>,
    /// Optional game index inside a suite.
    pub game_index: Option<usize>,
    /// Optional autoplay step index.
    pub step_index: Option<usize>,
}

impl OracleCaseProvenance {
    /// Creates provenance with a source label.
    pub fn new(source_label: impl Into<String>) -> Self {
        Self {
            source_label: source_label.into(),
            preset_name: None,
            seed: None,
            game_index: None,
            step_index: None,
        }
    }
}

/// Fully known deterministic state exported for external oracle validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleCase {
    /// Stable case id used to match local and external results.
    pub case_id: String,
    /// Full deterministic state to evaluate.
    pub full_state: FullState,
    /// Source/provenance metadata.
    pub provenance: OracleCaseProvenance,
    /// Optional expected/reference result bundled with the case.
    pub expected: Option<OracleReferenceResult>,
}

impl OracleCase {
    /// Creates an oracle case from a full deterministic state.
    pub fn new(
        case_id: impl Into<String>,
        full_state: FullState,
        provenance: OracleCaseProvenance,
    ) -> SolverResult<Self> {
        full_state.validate_consistency()?;
        Ok(Self {
            case_id: case_id.into(),
            full_state,
            provenance,
            expected: None,
        })
    }
}

/// Versioned collection of oracle cases.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleCasePack {
    /// Case-pack schema version.
    pub schema_version: String,
    /// Optional pack/source label.
    pub source_label: Option<String>,
    /// Cases in deterministic order.
    pub cases: Vec<OracleCase>,
}

impl OracleCasePack {
    /// Creates a versioned case pack.
    pub fn new(source_label: Option<String>, cases: Vec<OracleCase>) -> Self {
        Self {
            schema_version: ORACLE_CASE_SCHEMA_VERSION.to_string(),
            source_label,
            cases,
        }
    }

    /// Serializes the case pack as pretty JSON.
    pub fn to_json(&self) -> SolverResult<String> {
        to_pretty_json(self)
    }
}

/// Local deterministic solver mode used for oracle evaluation.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OracleEvaluationMode {
    /// Run proof-oriented deterministic search.
    Exact,
    /// Run bounded deterministic search.
    Bounded,
    /// Run fast deterministic value evaluation.
    Fast,
}

impl Default for OracleEvaluationMode {
    fn default() -> Self {
        Self::Exact
    }
}

/// Configuration for evaluating oracle cases with the local deterministic solver.
#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleEvaluationConfig {
    /// Solver search configuration.
    pub deterministic: DeterministicSearchConfig,
    /// Local evaluation mode.
    pub mode: OracleEvaluationMode,
}

impl Default for OracleEvaluationConfig {
    fn default() -> Self {
        Self {
            deterministic: DeterministicSearchConfig::default(),
            mode: OracleEvaluationMode::Exact,
        }
    }
}

/// External or local reference result in the simple interchange format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleReferenceResult {
    /// Case id.
    pub case_id: String,
    /// Reference outcome.
    pub outcome: SolveOutcome,
    /// Optional best move according to the reference.
    pub best_move: Option<MacroMoveKind>,
    /// Optional scalar value.
    pub value: Option<f32>,
    /// Optional reference/source label.
    pub source_label: Option<String>,
    /// Optional notes from the external tool or conversion script.
    pub notes: Option<String>,
}

/// Versioned collection of reference results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleReferenceResultSet {
    /// Result schema version.
    pub schema_version: String,
    /// Optional source label.
    pub source_label: Option<String>,
    /// Reference results in deterministic order.
    pub results: Vec<OracleReferenceResult>,
}

impl OracleReferenceResultSet {
    /// Creates a versioned reference result set.
    pub fn new(source_label: Option<String>, results: Vec<OracleReferenceResult>) -> Self {
        Self {
            schema_version: ORACLE_RESULT_SCHEMA_VERSION.to_string(),
            source_label,
            results,
        }
    }

    /// Serializes the result set as pretty JSON.
    pub fn to_json(&self) -> SolverResult<String> {
        to_pretty_json(self)
    }
}

/// Local deterministic evaluation result for one case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleCaseResult {
    /// Case id.
    pub case_id: String,
    /// Source/provenance metadata copied from the case.
    pub provenance: OracleCaseProvenance,
    /// Local solver outcome.
    pub outcome: SolveOutcome,
    /// Local best move, if any.
    pub best_move: Option<MacroMoveKind>,
    /// Local value or value estimate.
    pub value: f32,
    /// Local deterministic search stats.
    pub stats: DeterministicSearchStats,
}

impl OracleCaseResult {
    /// Converts the local result into the external-reference interchange shape.
    pub fn as_reference_result(&self) -> OracleReferenceResult {
        OracleReferenceResult {
            case_id: self.case_id.clone(),
            outcome: self.outcome,
            best_move: self.best_move,
            value: Some(self.value),
            source_label: Some("local_deterministic_solver".to_string()),
            notes: None,
        }
    }
}

/// Versioned local evaluation result pack.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleLocalEvaluation {
    /// Result schema version.
    pub schema_version: String,
    /// Evaluation mode used locally.
    pub mode: OracleEvaluationMode,
    /// Results in case order.
    pub results: Vec<OracleCaseResult>,
}

impl OracleLocalEvaluation {
    /// Serializes local evaluation as pretty JSON.
    pub fn to_json(&self) -> SolverResult<String> {
        to_pretty_json(self)
    }

    /// Converts all local results to reference interchange rows.
    pub fn as_reference_results(&self) -> Vec<OracleReferenceResult> {
        self.results
            .iter()
            .map(OracleCaseResult::as_reference_result)
            .collect()
    }
}

/// Mismatch category for local/reference comparisons.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OracleMismatchKind {
    /// Local result has no matching reference row.
    MissingReference,
    /// Reference row has no matching local result.
    MissingLocal,
    /// Both sides are exact win/loss results but disagree.
    OutcomeMismatch,
    /// One side is exact and the other is unknown.
    AmbiguousComparison,
    /// Outcomes are compatible but best moves differ.
    BestMoveMismatch,
}

/// One local/reference comparison row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleComparisonRecord {
    /// Case id.
    pub case_id: String,
    /// Local result, if present.
    pub local: Option<OracleReferenceResult>,
    /// Reference result, if present.
    pub reference: Option<OracleReferenceResult>,
    /// Mismatch category, if any.
    pub mismatch_kind: Option<OracleMismatchKind>,
    /// Whether outcomes match exactly when both sides are present.
    pub outcome_matches: Option<bool>,
    /// Whether best moves match when both sides provided a best move.
    pub best_move_matches: Option<bool>,
}

/// Count of one mismatch category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OracleMismatchCount {
    /// Mismatch kind.
    pub kind: OracleMismatchKind,
    /// Number of records with this kind.
    pub count: usize,
}

/// Summary for an oracle comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleComparisonSummary {
    /// Number of ids in the local/reference union.
    pub cases_compared: usize,
    /// Number of records with no mismatch.
    pub matches: usize,
    /// Number of records with any mismatch.
    pub mismatches: usize,
    /// Exact win/loss agreements.
    pub exact_win_loss_agreements: usize,
    /// Best-move agreements among rows where both sides provided a move.
    pub best_move_agreements: usize,
    /// Number of rows where both sides provided a best move.
    pub best_move_comparisons: usize,
    /// Per-category mismatch counts.
    pub mismatch_counts: Vec<OracleMismatchCount>,
    /// Per-case comparison records.
    pub records: Vec<OracleComparisonRecord>,
}

impl OracleComparisonSummary {
    /// Exports a deterministic JSON summary.
    pub fn to_json_summary(&self) -> SolverResult<String> {
        to_pretty_json(self)
    }

    /// Exports a deterministic CSV summary with one row per case.
    pub fn to_csv_summary(&self) -> String {
        let rows = self
            .records
            .iter()
            .map(|record| {
                vec![
                    record.case_id.clone(),
                    record
                        .local
                        .as_ref()
                        .map(|result| format!("{:?}", result.outcome))
                        .unwrap_or_default(),
                    record
                        .reference
                        .as_ref()
                        .map(|result| format!("{:?}", result.outcome))
                        .unwrap_or_default(),
                    record
                        .local
                        .as_ref()
                        .and_then(|result| result.best_move)
                        .map(|best| format!("{best:?}"))
                        .unwrap_or_default(),
                    record
                        .reference
                        .as_ref()
                        .and_then(|result| result.best_move)
                        .map(|best| format!("{best:?}"))
                        .unwrap_or_default(),
                    record
                        .mismatch_kind
                        .map(|kind| format!("{kind:?}"))
                        .unwrap_or_default(),
                    record
                        .outcome_matches
                        .map(|matches| matches.to_string())
                        .unwrap_or_default(),
                    record
                        .best_move_matches
                        .map(|matches| matches.to_string())
                        .unwrap_or_default(),
                ]
            })
            .collect::<Vec<_>>();
        csv_table(
            &[
                "case_id",
                "local_outcome",
                "reference_outcome",
                "local_best_move",
                "reference_best_move",
                "mismatch_kind",
                "outcome_matches",
                "best_move_matches",
            ],
            &rows,
        )
    }
}

/// Generates one oracle case per seeded full deal in a benchmark suite.
pub fn oracle_cases_from_seeded_suite(
    suite: &BenchmarkSuite,
    preset: Option<&ExperimentPreset>,
) -> SolverResult<OracleCasePack> {
    let runner = ExperimentRunner;
    let mut cases = Vec::with_capacity(suite.seeds.len());
    for (index, seed) in suite.seeds.iter().copied().enumerate() {
        let deal = runner.generate_deal(seed)?;
        let mut provenance = OracleCaseProvenance::new("seeded_suite");
        provenance.seed = Some(seed);
        provenance.game_index = Some(index);
        provenance.preset_name = preset.map(|preset| preset.name.clone());
        let case_id = format!("{}:seed:{}:case:{}", suite.name, seed.0, index);
        cases.push(OracleCase::new(case_id, deal.full_state, provenance)?);
    }
    Ok(OracleCasePack::new(Some(suite.name.clone()), cases))
}

/// Builds oracle cases from an explicit list of full deterministic states.
pub fn oracle_cases_from_full_states(
    source_label: impl Into<String>,
    states: impl IntoIterator<Item = FullState>,
) -> SolverResult<OracleCasePack> {
    let source_label = source_label.into();
    let mut cases = Vec::new();
    for (index, full_state) in states.into_iter().enumerate() {
        let provenance = OracleCaseProvenance {
            source_label: source_label.clone(),
            preset_name: None,
            seed: None,
            game_index: None,
            step_index: None,
        };
        cases.push(OracleCase::new(
            format!("{source_label}:manual:{index}"),
            full_state,
            provenance,
        )?);
    }
    Ok(OracleCasePack::new(Some(source_label), cases))
}

/// Reconstructs full-state decision cases from an autoplay trace.
pub fn oracle_cases_from_autoplay_trace(
    source_label: impl Into<String>,
    initial_full_state: &FullState,
    trace: &AutoplayTrace,
    seed: Option<DealSeed>,
    preset_name: Option<String>,
) -> SolverResult<OracleCasePack> {
    let source_label = source_label.into();
    let mut state = initial_full_state.clone();
    state.validate_consistency()?;
    let mut cases = Vec::with_capacity(trace.steps.len().saturating_add(1));

    for step_index in 0..=trace.steps.len() {
        let mut provenance = OracleCaseProvenance::new(source_label.clone());
        provenance.seed = seed;
        provenance.game_index = Some(0);
        provenance.step_index = Some(step_index);
        provenance.preset_name = preset_name.clone();
        cases.push(OracleCase::new(
            format!("{source_label}:step:{step_index}"),
            state.clone(),
            provenance,
        )?);

        let Some(step) = trace.steps.get(step_index) else {
            break;
        };
        apply_atomic_move_full_state(&mut state, step.chosen_move.atomic)?;
    }

    Ok(OracleCasePack::new(Some(source_label), cases))
}

/// Evaluates oracle cases with the local deterministic solver.
pub fn evaluate_oracle_cases(
    cases: &[OracleCase],
    config: OracleEvaluationConfig,
) -> SolverResult<OracleLocalEvaluation> {
    let solver = DeterministicSolver::new(config.deterministic);
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        case.full_state.validate_consistency()?;
        let (outcome, best_move, value, stats) = match config.mode {
            OracleEvaluationMode::Exact => {
                let result = solver.solve_exact(&case.full_state)?;
                (
                    result.outcome,
                    result.best_move.map(|best| best.kind),
                    result.value,
                    result.stats,
                )
            }
            OracleEvaluationMode::Bounded => {
                let result = solver.solve_bounded(&case.full_state)?;
                (
                    result.outcome,
                    result.best_move.map(|best| best.kind),
                    result.estimated_value,
                    result.stats,
                )
            }
            OracleEvaluationMode::Fast => {
                let result = solver.evaluate_fast(&case.full_state)?;
                (
                    if case.full_state.visible.is_structural_win() {
                        SolveOutcome::ProvenWin
                    } else {
                        SolveOutcome::Unknown
                    },
                    result.best_move.map(|best| best.kind),
                    result.value,
                    result.stats,
                )
            }
        };
        results.push(OracleCaseResult {
            case_id: case.case_id.clone(),
            provenance: case.provenance.clone(),
            outcome,
            best_move,
            value,
            stats,
        });
    }
    Ok(OracleLocalEvaluation {
        schema_version: ORACLE_RESULT_SCHEMA_VERSION.to_string(),
        mode: config.mode,
        results,
    })
}

/// Compares local solver results with external/reference results by case id.
pub fn compare_oracle_results(
    local: &[OracleCaseResult],
    reference: &[OracleReferenceResult],
) -> OracleComparisonSummary {
    let local_map = local
        .iter()
        .map(|result| (result.case_id.clone(), result.as_reference_result()))
        .collect::<BTreeMap<_, _>>();
    compare_oracle_reference_results(&local_map, reference)
}

/// Compares two reference-result collections by case id.
pub fn compare_oracle_reference_results(
    local: &BTreeMap<String, OracleReferenceResult>,
    reference: &[OracleReferenceResult],
) -> OracleComparisonSummary {
    let reference_map = reference
        .iter()
        .map(|result| (result.case_id.clone(), result.clone()))
        .collect::<BTreeMap<_, _>>();
    let case_ids = local
        .keys()
        .chain(reference_map.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut records = Vec::with_capacity(case_ids.len());
    let mut exact_win_loss_agreements = 0usize;
    let mut best_move_agreements = 0usize;
    let mut best_move_comparisons = 0usize;

    for case_id in case_ids {
        let local_result = local.get(&case_id).cloned();
        let reference_result = reference_map.get(&case_id).cloned();
        let (mismatch_kind, outcome_matches, best_move_matches) =
            classify_oracle_comparison(local_result.as_ref(), reference_result.as_ref());

        if local_result
            .as_ref()
            .zip(reference_result.as_ref())
            .is_some_and(|(local, reference)| {
                local.outcome == reference.outcome && is_exact_win_loss(local.outcome)
            })
        {
            exact_win_loss_agreements += 1;
        }
        if let Some((local_best, reference_best)) =
            local_result.as_ref().and_then(|local| local.best_move).zip(
                reference_result
                    .as_ref()
                    .and_then(|reference| reference.best_move),
            )
        {
            best_move_comparisons += 1;
            if local_best == reference_best {
                best_move_agreements += 1;
            }
        }

        records.push(OracleComparisonRecord {
            case_id,
            local: local_result,
            reference: reference_result,
            mismatch_kind,
            outcome_matches,
            best_move_matches,
        });
    }

    let matches = records
        .iter()
        .filter(|record| record.mismatch_kind.is_none())
        .count();
    let mismatches = records.len().saturating_sub(matches);
    let mismatch_counts = summarize_oracle_mismatches(&records);

    OracleComparisonSummary {
        cases_compared: records.len(),
        matches,
        mismatches,
        exact_win_loss_agreements,
        best_move_agreements,
        best_move_comparisons,
        mismatch_counts,
        records,
    }
}

/// Saves a case pack as JSON.
pub fn save_oracle_case_pack(path: impl AsRef<Path>, pack: &OracleCasePack) -> SolverResult<()> {
    fs::write(path, pack.to_json()?)?;
    Ok(())
}

/// Loads a JSON oracle case pack.
pub fn load_oracle_case_pack(path: impl AsRef<Path>) -> SolverResult<OracleCasePack> {
    let contents = fs::read_to_string(path)?;
    serde_json::from_str(&contents).map_err(|error| SolverError::Serialization(error.to_string()))
}

/// Saves local evaluation results as JSON.
pub fn save_oracle_local_evaluation(
    path: impl AsRef<Path>,
    evaluation: &OracleLocalEvaluation,
) -> SolverResult<()> {
    fs::write(path, evaluation.to_json()?)?;
    Ok(())
}

/// Loads local evaluation results as JSON.
pub fn load_oracle_local_evaluation(path: impl AsRef<Path>) -> SolverResult<OracleLocalEvaluation> {
    let contents = fs::read_to_string(path)?;
    serde_json::from_str(&contents).map_err(|error| SolverError::Serialization(error.to_string()))
}

/// Loads external reference results from JSON set, JSON array, local-eval JSON, or JSONL.
pub fn load_oracle_reference_results(
    path: impl AsRef<Path>,
) -> SolverResult<Vec<OracleReferenceResult>> {
    let contents = fs::read_to_string(path)?;
    parse_oracle_reference_results(&contents)
}

/// Parses external reference results from JSON set, JSON array, local-eval JSON, or JSONL.
pub fn parse_oracle_reference_results(text: &str) -> SolverResult<Vec<OracleReferenceResult>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    if trimmed.starts_with('{') {
        if let Ok(set) = serde_json::from_str::<OracleReferenceResultSet>(trimmed) {
            return Ok(set.results);
        }
        if let Ok(local) = serde_json::from_str::<OracleLocalEvaluation>(trimmed) {
            return Ok(local.as_reference_results());
        }
    }
    if trimmed.starts_with('[') {
        return serde_json::from_str::<Vec<OracleReferenceResult>>(trimmed)
            .map_err(|error| SolverError::Serialization(error.to_string()));
    }

    trimmed
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<OracleReferenceResult>(line)
                .map_err(|error| SolverError::Serialization(error.to_string()))
        })
        .collect()
}

fn classify_oracle_comparison(
    local: Option<&OracleReferenceResult>,
    reference: Option<&OracleReferenceResult>,
) -> (Option<OracleMismatchKind>, Option<bool>, Option<bool>) {
    let Some(local) = local else {
        return (
            Some(OracleMismatchKind::MissingLocal),
            None,
            reference
                .and_then(|reference| reference.best_move)
                .map(|_| false),
        );
    };
    let Some(reference) = reference else {
        return (
            Some(OracleMismatchKind::MissingReference),
            None,
            local.best_move.map(|_| false),
        );
    };

    let outcome_matches = local.outcome == reference.outcome;
    let best_move_matches = local
        .best_move
        .zip(reference.best_move)
        .map(|(left, right)| left == right);

    let mismatch_kind = if is_exact_win_loss(local.outcome)
        && is_exact_win_loss(reference.outcome)
        && !outcome_matches
    {
        Some(OracleMismatchKind::OutcomeMismatch)
    } else if local.outcome != reference.outcome
        && (local.outcome == SolveOutcome::Unknown || reference.outcome == SolveOutcome::Unknown)
    {
        Some(OracleMismatchKind::AmbiguousComparison)
    } else if best_move_matches == Some(false) {
        Some(OracleMismatchKind::BestMoveMismatch)
    } else {
        None
    };

    (mismatch_kind, Some(outcome_matches), best_move_matches)
}

fn summarize_oracle_mismatches(records: &[OracleComparisonRecord]) -> Vec<OracleMismatchCount> {
    let mut counts = BTreeMap::<OracleMismatchKind, usize>::new();
    for kind in records.iter().filter_map(|record| record.mismatch_kind) {
        *counts.entry(kind).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .map(|(kind, count)| OracleMismatchCount { kind, count })
        .collect()
}

const fn is_exact_win_loss(outcome: SolveOutcome) -> bool {
    matches!(outcome, SolveOutcome::ProvenWin | SolveOutcome::ProvenLoss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cards::Suit, core::FoundationState};

    fn won_state() -> FullState {
        let mut visible = crate::core::VisibleState::default();
        let mut foundations = FoundationState::default();
        for suit in [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
            foundations.set_top_rank(suit, Some(crate::cards::Rank::King));
        }
        visible.foundations = foundations;
        FullState::new(visible, crate::core::HiddenAssignments::empty())
    }

    #[test]
    fn oracle_case_export_is_reproducible() {
        let suite = BenchmarkSuite::from_base_seed("oracle", 77, 2);
        let first = oracle_cases_from_seeded_suite(&suite, None).unwrap();
        let second = oracle_cases_from_seeded_suite(&suite, None).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.schema_version, ORACLE_CASE_SCHEMA_VERSION);
        assert_eq!(first.cases.len(), 2);
        assert!(first.to_json().unwrap().contains("seeded_suite"));
    }

    #[test]
    fn local_evaluation_records_solver_shape() {
        let pack = oracle_cases_from_full_states("manual", [won_state()]).unwrap();
        let evaluation = evaluate_oracle_cases(
            &pack.cases,
            OracleEvaluationConfig {
                deterministic: DeterministicSearchConfig {
                    budget: crate::deterministic_solver::SolveBudget {
                        node_budget: Some(10),
                        depth_budget: Some(1),
                        wall_clock_limit_ms: None,
                    },
                    ..DeterministicSearchConfig::default()
                },
                mode: OracleEvaluationMode::Exact,
            },
        )
        .unwrap();

        assert_eq!(evaluation.results.len(), 1);
        assert_eq!(evaluation.results[0].outcome, SolveOutcome::ProvenWin);
        assert_eq!(
            evaluation.as_reference_results()[0].case_id,
            "manual:manual:0"
        );
    }

    #[test]
    fn comparison_summary_counts_matches_and_mismatches() {
        let local = vec![
            OracleCaseResult {
                case_id: "a".to_string(),
                provenance: OracleCaseProvenance::new("test"),
                outcome: SolveOutcome::ProvenWin,
                best_move: None,
                value: 1.0,
                stats: DeterministicSearchStats::default(),
            },
            OracleCaseResult {
                case_id: "b".to_string(),
                provenance: OracleCaseProvenance::new("test"),
                outcome: SolveOutcome::ProvenLoss,
                best_move: None,
                value: 0.0,
                stats: DeterministicSearchStats::default(),
            },
        ];
        let reference = vec![
            OracleReferenceResult {
                case_id: "a".to_string(),
                outcome: SolveOutcome::ProvenWin,
                best_move: None,
                value: Some(1.0),
                source_label: Some("oracle".to_string()),
                notes: None,
            },
            OracleReferenceResult {
                case_id: "b".to_string(),
                outcome: SolveOutcome::ProvenWin,
                best_move: None,
                value: Some(1.0),
                source_label: Some("oracle".to_string()),
                notes: None,
            },
            OracleReferenceResult {
                case_id: "c".to_string(),
                outcome: SolveOutcome::Unknown,
                best_move: None,
                value: None,
                source_label: Some("oracle".to_string()),
                notes: None,
            },
        ];

        let summary = compare_oracle_results(&local, &reference);

        assert_eq!(summary.cases_compared, 3);
        assert_eq!(summary.matches, 1);
        assert_eq!(summary.mismatches, 2);
        assert_eq!(summary.exact_win_loss_agreements, 1);
        assert!(summary
            .mismatch_counts
            .iter()
            .any(|count| count.kind == OracleMismatchKind::OutcomeMismatch && count.count == 1));
        assert!(summary
            .mismatch_counts
            .iter()
            .any(|count| count.kind == OracleMismatchKind::MissingLocal && count.count == 1));
        assert!(summary.to_csv_summary().contains("OutcomeMismatch"));
    }

    #[test]
    fn reference_jsonl_ingestion_is_supported() {
        let line = serde_json::to_string(&OracleReferenceResult {
            case_id: "x".to_string(),
            outcome: SolveOutcome::Unknown,
            best_move: None,
            value: None,
            source_label: None,
            notes: None,
        })
        .unwrap();

        let parsed = parse_oracle_reference_results(&line).unwrap();

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].case_id, "x");
    }
}
