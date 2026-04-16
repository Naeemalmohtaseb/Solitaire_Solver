//! PIMC baseline, full-game autoplay, seeded benchmarks, reporting, and presets.
//!
//! The experiment layer owns reproducible deal suites, honest baseline evaluation,
//! full-game autoplay orchestration, paired A/B comparisons, and machine-friendly
//! reports. It does not introduce weighted hidden-card posteriors or new planner
//! logic.

mod autoplay;
mod benchmark;
mod calibration;
mod pimc;
mod presets;
mod reporting;

pub(crate) use autoplay::recommend_autoplay_move;
pub use autoplay::*;
pub use benchmark::*;
pub use calibration::*;
pub use pimc::*;
pub use presets::*;
pub use reporting::*;

use serde::Serialize;

use crate::{
    cards::Card,
    config::{DeterministicSolverConfig, SolverConfig},
    core::{
        BeliefState, FullState, HiddenAssignment, HiddenAssignments, HiddenSlot, TableauColumn,
        UnseenCardSet, VisibleState,
    },
    deterministic_solver::{
        DeterministicSearchConfig, DeterministicTtConfig, EvaluatorWeights, SolveBudget,
    },
    error::{SolverError, SolverResult},
    stock::CyclicStockState,
    types::{ColumnId, DealSeed},
};
fn deterministic_config_with_override(
    mut deterministic: DeterministicSearchConfig,
    pimc: PimcConfig,
) -> DeterministicSearchConfig {
    if let Some(node_budget) = pimc.per_world_node_budget_override {
        deterministic.budget.node_budget = Some(node_budget);
    }
    deterministic
}

pub(crate) fn deterministic_search_config_from_solver(
    solver_config: &SolverConfig,
) -> DeterministicSearchConfig {
    deterministic_search_config_from_parts(&solver_config.deterministic, solver_config)
}

fn deterministic_search_config_from_parts(
    deterministic: &DeterministicSolverConfig,
    solver_config: &SolverConfig,
) -> DeterministicSearchConfig {
    DeterministicSearchConfig {
        budget: SolveBudget {
            node_budget: Some(deterministic.fast_eval_node_budget),
            depth_budget: Some(deterministic.max_macro_depth),
            wall_clock_limit_ms: None,
        },
        closure: solver_config.closure,
        allow_foundation_retreats: deterministic.enable_foundation_retreats,
        evaluator_weights: EvaluatorWeights::default(),
        tt: DeterministicTtConfig {
            enabled: deterministic.enable_tt,
            capacity: deterministic.tt_capacity,
            store_approx: deterministic.tt_store_approx,
        },
    }
}

fn summarize_benchmark(
    config: EngineConfigLabel,
    suite: &BenchmarkSuite,
    records: Vec<BenchmarkRecord>,
) -> BenchmarkResult {
    let deals = records.len();
    let wins = records.iter().filter(|record| record.win).count();
    let losses = deals.saturating_sub(wins);
    let win_rate = if deals == 0 {
        0.0
    } else {
        wins as f64 / deals as f64
    };
    let mean_time_ms = mean(records.iter().map(|record| record.elapsed_ms as f64));
    let mean_nodes = mean(
        records
            .iter()
            .map(|record| record.deterministic_nodes as f64),
    );
    let mean_samples = mean(records.iter().map(|record| record.sample_count as f64));

    BenchmarkResult {
        config,
        suite_name: suite.name.clone(),
        suite: suite.description(),
        records,
        deals,
        wins,
        losses,
        win_rate,
        mean_time_ms,
        mean_nodes,
        mean_samples,
    }
}

fn summarize_autoplay_benchmark(
    config: EngineConfigLabel,
    backend: PlannerBackend,
    suite: &BenchmarkSuite,
    records: Vec<AutoplayBenchmarkRecord>,
) -> AutoplayBenchmarkResult {
    let games = records.len();
    let wins = records.iter().filter(|record| record.won).count();
    let losses = games.saturating_sub(wins);
    let win_rate = if games == 0 {
        0.0
    } else {
        wins as f64 / games as f64
    };
    let average_moves_per_game = mean(records.iter().map(|record| record.moves_played as f64));
    let average_planner_time_per_move_ms = mean(
        records
            .iter()
            .map(|record| record.mean_planner_time_per_move_ms),
    );
    let average_total_planner_time_per_game_ms = mean(
        records
            .iter()
            .map(|record| record.total_planner_time_ms as f64),
    );
    let average_deterministic_nodes = mean(
        records
            .iter()
            .map(|record| record.deterministic_nodes as f64),
    );
    let average_root_visits = mean(records.iter().map(|record| record.root_visits as f64));
    let late_exact_trigger_count = records
        .iter()
        .map(|record| record.late_exact_triggers)
        .sum::<usize>();
    let terminations = summarize_terminations(&records);

    AutoplayBenchmarkResult {
        config,
        backend,
        suite_name: suite.name.clone(),
        suite: suite.description(),
        records,
        games,
        wins,
        losses,
        win_rate,
        average_moves_per_game,
        average_planner_time_per_move_ms,
        average_total_planner_time_per_game_ms,
        average_deterministic_nodes,
        average_root_visits,
        late_exact_trigger_count,
        terminations,
    }
}

fn summarize_terminations(records: &[AutoplayBenchmarkRecord]) -> Vec<AutoplayTerminationCount> {
    let mut counts = Vec::<AutoplayTerminationCount>::new();
    for record in records {
        match counts
            .iter_mut()
            .find(|entry| entry.termination == record.termination)
        {
            Some(entry) => entry.count += 1,
            None => counts.push(AutoplayTerminationCount {
                termination: record.termination,
                count: 1,
            }),
        }
    }
    counts.sort_by_key(|entry| termination_sort_key(entry.termination));
    counts
}

const fn termination_sort_key(termination: AutoplayTermination) -> u8 {
    match termination {
        AutoplayTermination::Win => 0,
        AutoplayTermination::NoLegalMove => 1,
        AutoplayTermination::StepLimit => 2,
        AutoplayTermination::BudgetExhausted => 3,
    }
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn standard_error(values: impl Iterator<Item = f64>) -> f64 {
    let values = values.collect::<Vec<_>>();
    if values.len() <= 1 {
        return 0.0;
    }
    let mean_value = mean(values.iter().copied());
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean_value;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (variance / values.len() as f64).sqrt()
}

fn to_pretty_json<T: Serialize>(value: &T) -> SolverResult<String> {
    serde_json::to_string_pretty(value)
        .map_err(|error| SolverError::Serialization(error.to_string()))
}

fn csv_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let mut output = String::new();
    output.push_str(
        &headers
            .iter()
            .map(|header| csv_escape(header))
            .collect::<Vec<_>>()
            .join(","),
    );
    output.push('\n');
    for row in rows {
        output.push_str(
            &row.iter()
                .map(|field| csv_escape(field))
                .collect::<Vec<_>>()
                .join(","),
        );
        output.push('\n');
    }
    output
}

fn csv_escape(value: impl AsRef<str>) -> String {
    let value = value.as_ref();
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn optional_seed_string(seed: Option<DealSeed>) -> String {
    seed.map(|seed| seed.0.to_string()).unwrap_or_default()
}

fn termination_counts_string(counts: &[AutoplayTerminationCount]) -> String {
    counts
        .iter()
        .map(|entry| format!("{:?}:{}", entry.termination, entry.count))
        .collect::<Vec<_>>()
        .join(";")
}

fn action_seed(seed: DealSeed, action_index: usize) -> DealSeed {
    DealSeed(seed.0 ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(action_index as u64 + 1)))
}

fn generate_benchmark_deal(seed: DealSeed) -> SolverResult<BenchmarkDeal> {
    let mut deck = (0..Card::COUNT)
        .map(|index| Card::new(index as u8))
        .collect::<SolverResult<Vec<_>>>()?;
    let mut rng = ExperimentRng::new(seed.0);
    shuffle_cards(&mut deck, &mut rng);

    let mut cursor = 0usize;
    let mut columns = std::array::from_fn(|_| TableauColumn::default());
    let mut hidden_assignments = Vec::new();
    let mut unseen_cards = Vec::new();

    for (column_index, column) in columns.iter_mut().enumerate() {
        let hidden_count = column_index as u8;
        let column_id = ColumnId::new(column_index as u8)?;
        let hidden_start = cursor;
        let hidden_end = cursor + column_index;
        let face_up = deck[hidden_end];
        cursor = hidden_end + 1;

        *column = TableauColumn::new(hidden_count, vec![face_up]);
        for depth in 0..hidden_count {
            let card = deck[hidden_start + usize::from(depth)];
            unseen_cards.push(card);
            hidden_assignments.push(HiddenAssignment::new(
                HiddenSlot::new(column_id, depth),
                card,
            ));
        }
    }

    let stock_cards = deck[cursor..].to_vec();
    let visible = VisibleState {
        foundations: crate::core::FoundationState::default(),
        columns,
        stock: CyclicStockState::new(stock_cards, None, 0, None, 3),
    };
    let full_state = FullState::new(visible.clone(), HiddenAssignments::new(hidden_assignments));
    let belief_state = BeliefState::new(visible, UnseenCardSet::from_cards(unseen_cards)?);

    full_state.validate_consistency()?;
    belief_state.validate_consistency_against_visible()?;

    Ok(BenchmarkDeal {
        seed,
        full_state,
        belief_state,
    })
}

fn shuffle_cards(cards: &mut [Card], rng: &mut ExperimentRng) {
    for index in (1..cards.len()).rev() {
        let swap_index = rng.next_bounded(index + 1);
        cards.swap(index, swap_index);
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct ExperimentRng {
    state: u64,
}

impl ExperimentRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_bounded(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u64() % upper_exclusive as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        belief::{validate_belief_against_full_state, validate_sample_against_belief},
        cards::{Rank, Suit},
        core::{FoundationState, HiddenSlot},
        deterministic_solver::{ordered_macro_moves, SolveOutcome},
        late_exact::LateExactEvaluationMode,
        moves::MacroMoveKind,
        planner::{BeliefPlannerConfig, PlannerLeafEvalMode},
    };

    fn fast_pimc_config(seed: u64) -> PimcConfig {
        PimcConfig {
            sample_count: 4,
            deterministic_mode: PimcEvaluationMode::Fast,
            per_world_node_budget_override: Some(16),
            shared_world_batch: true,
            rng_seed: DealSeed(seed),
            max_candidate_actions: Some(3),
            report_standard_error: true,
        }
    }

    fn fast_benchmark_config(name: &str, seed: u64) -> BenchmarkConfig {
        BenchmarkConfig {
            label: EngineConfigLabel::new(name),
            pimc: fast_pimc_config(seed),
            deterministic: DeterministicSearchConfig {
                budget: crate::deterministic_solver::SolveBudget {
                    node_budget: Some(16),
                    depth_budget: Some(1),
                    wall_clock_limit_ms: None,
                },
                ..DeterministicSearchConfig::default()
            },
        }
    }

    fn fast_solver_config() -> SolverConfig {
        let mut config = SolverConfig::default();
        config.deterministic.fast_eval_node_budget = 16;
        config.deterministic.exact_node_budget = 32;
        config.deterministic.max_macro_depth = 2;
        config.deterministic.enable_tt = false;
        config.deterministic.enable_foundation_retreats = false;
        config.belief_planner = BeliefPlannerConfig {
            simulation_budget: 2,
            max_depth: 1,
            exploration_constant: 1.0,
            leaf_world_samples: 1,
            leaf_eval_mode: PlannerLeafEvalMode::Fast,
            rng_seed: DealSeed(1),
            enable_early_stop: false,
            initial_screen_simulations: 0,
            max_active_root_actions: None,
            enable_second_reveal_refinement: false,
            ..BeliefPlannerConfig::default()
        };
        config.late_exact.enabled = true;
        config.late_exact.hidden_card_threshold = 8;
        config.late_exact.max_root_actions = 1;
        config.late_exact.evaluation_mode = LateExactEvaluationMode::Fast;
        config
    }

    fn fast_autoplay_config(backend: PlannerBackend) -> AutoplayConfig {
        AutoplayConfig {
            backend,
            pimc: fast_pimc_config(1),
            max_steps: 8,
            max_total_planner_time_ms: None,
            validate_each_step: true,
        }
    }

    fn fast_autoplay_benchmark_config(
        name: &str,
        backend: PlannerBackend,
    ) -> AutoplayBenchmarkConfig {
        let mut autoplay = fast_autoplay_config(backend);
        autoplay.max_steps = 0;
        AutoplayBenchmarkConfig {
            label: EngineConfigLabel::new(name),
            solver: fast_solver_config(),
            autoplay,
        }
    }

    fn all_stock_belief() -> BeliefState {
        let stock_cards = (0..Card::COUNT)
            .map(|index| Card::new(index as u8).unwrap())
            .collect();
        let visible = VisibleState {
            foundations: crate::core::FoundationState::default(),
            columns: std::array::from_fn(|_| TableauColumn::default()),
            stock: CyclicStockState::new(stock_cards, None, 0, None, 3),
        };
        BeliefState::new(visible, UnseenCardSet::empty())
    }

    fn complete_foundations() -> FoundationState {
        let mut foundations = FoundationState::default();
        for suit in [Suit::Clubs, Suit::Diamonds, Suit::Hearts, Suit::Spades] {
            foundations.set_top_rank(suit, Some(Rank::King));
        }
        foundations
    }

    fn won_full_state() -> FullState {
        let mut visible = VisibleState::default();
        visible.foundations = complete_foundations();
        FullState::new(visible, HiddenAssignments::empty())
    }

    fn no_move_full_state() -> FullState {
        let cards = (0..Card::COUNT)
            .map(|index| Card::new(index as u8).unwrap())
            .collect();
        let mut visible = VisibleState::default();
        visible.stock = CyclicStockState::from_parts(cards, 0, 0, 0, Some(0), 3);
        FullState::new(visible, HiddenAssignments::empty())
    }

    fn forced_reveal_full_state() -> FullState {
        let mut visible = VisibleState::default();
        visible.foundations = complete_foundations();
        visible
            .foundations
            .set_top_rank(Suit::Spades, Some(Rank::Jack));
        visible.columns[0] = TableauColumn::new(1, vec!["Qs".parse().unwrap()]);
        FullState::new(
            visible,
            HiddenAssignments::new(vec![HiddenAssignment::new(
                HiddenSlot::new(ColumnId::new(0).unwrap(), 0),
                "Ks".parse().unwrap(),
            )]),
        )
    }

    #[test]
    fn same_belief_and_seed_give_reproducible_world_batches() {
        let deal = ExperimentRunner.generate_deal(DealSeed(42)).unwrap();

        let first = PimcWorldBatch::sample(&deal.belief_state, 5, DealSeed(99)).unwrap();
        let second = PimcWorldBatch::sample(&deal.belief_state, 5, DealSeed(99)).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn shared_world_batch_is_reused_across_candidate_actions() {
        let deal = ExperimentRunner.generate_deal(DealSeed(7)).unwrap();
        let config = fast_pimc_config(123);

        let recommendation = recommend_move_pimc(
            &deal.belief_state,
            DeterministicSearchConfig::default(),
            config,
        )
        .unwrap();

        let batch_len = recommendation.world_batch.as_ref().unwrap().samples.len();
        assert_eq!(batch_len, config.sample_count);
        assert!(recommendation
            .action_stats
            .iter()
            .all(|stats| stats.visits == batch_len));
    }

    #[test]
    fn pimc_recommendation_returns_only_legal_move_immediately() {
        let belief = all_stock_belief();

        let recommendation = recommend_move_pimc(
            &belief,
            DeterministicSearchConfig::default(),
            fast_pimc_config(1),
        )
        .unwrap();

        assert!(recommendation.best_move.is_some());
        assert_eq!(recommendation.sample_count, 0);
        assert!(recommendation.world_batch.is_none());
    }

    #[test]
    fn per_action_statistics_aggregate_deterministically() {
        let mut stats = PimcActionStats::new(
            ordered_macro_moves(
                &all_stock_belief().visible,
                DeterministicSearchConfig::default(),
            )[0]
            .clone(),
        );

        stats.record(1.0, SolveOutcome::ProvenWin, 10);
        stats.record(0.0, SolveOutcome::ProvenLoss, 20);

        assert_eq!(stats.visits, 2);
        assert_eq!(stats.exact_wins, 1);
        assert_eq!(stats.exact_losses, 1);
        assert!((stats.mean_value - 0.5).abs() < f64::EPSILON);
        assert_eq!(stats.deterministic_nodes, 30);
    }

    #[test]
    fn ab_experiments_use_same_seed_suite_for_both_configs() {
        let runner = ExperimentRunner;
        let suite = BenchmarkSuite::from_seed_range("tiny", 10, 2);
        let baseline = fast_benchmark_config("A", 1);
        let candidate = fast_benchmark_config("B", 2);

        let result = runner
            .run_paired_comparison(&suite, &baseline, &candidate)
            .unwrap();

        let baseline_seeds: Vec<_> = result
            .baseline
            .records
            .iter()
            .map(|record| record.seed)
            .collect();
        let candidate_seeds: Vec<_> = result
            .candidate
            .records
            .iter()
            .map(|record| record.seed)
            .collect();

        assert_eq!(baseline_seeds, candidate_seeds);
        assert_eq!(baseline_seeds, suite.seeds);
    }

    #[test]
    fn repeated_suite_runner_preserves_seed_reproducibility() {
        let runner = ExperimentRunner;
        let baseline = fast_benchmark_config("A", 1);
        let candidate = fast_benchmark_config("B", 2);

        let first = runner
            .run_repeated_comparison("rep", 100, 2, 2, &baseline, &candidate)
            .unwrap();
        let second = runner
            .run_repeated_comparison("rep", 100, 2, 2, &baseline, &candidate)
            .unwrap();

        let first_seeds: Vec<_> = first
            .repetitions
            .iter()
            .map(|summary| summary.suite.seeds.clone())
            .collect();
        let second_seeds: Vec<_> = second
            .repetitions
            .iter()
            .map(|summary| summary.suite.seeds.clone())
            .collect();

        assert_eq!(first_seeds, second_seeds);
    }

    #[test]
    fn benchmark_suite_helpers_are_reproducible() {
        let explicit = BenchmarkSuite::from_seeds("fixed", [DealSeed(1), DealSeed(3)]);
        assert_eq!(explicit.seeds, vec![DealSeed(1), DealSeed(3)]);
        assert_eq!(explicit.description().base_seed, None);

        let first = BenchmarkSuite::repeated_from_base_seed("rep", 900, 2, 2);
        let second = BenchmarkSuite::repeated_from_base_seed("rep", 900, 2, 2);

        assert_eq!(first, second);
        assert_eq!(first[0].description().base_seed, Some(DealSeed(900)));
        assert_eq!(first[1].description().base_seed, Some(DealSeed(1_000_900)));
    }

    #[test]
    fn preset_constructors_are_reproducible_and_named() {
        let first = fast_benchmark();
        let second = fast_benchmark();

        assert_eq!(first, second);
        assert_eq!(first.name, "fast_benchmark");
        assert_eq!(pimc_baseline().autoplay.backend, PlannerBackend::Pimc);
        assert_eq!(
            belief_uct_default().autoplay.backend,
            PlannerBackend::BeliefUct
        );
        assert_eq!(
            belief_uct_late_exact().autoplay.backend,
            PlannerBackend::BeliefUctLateExact
        );
        assert_eq!(
            balanced_benchmark().autoplay.backend,
            PlannerBackend::BeliefUctLateExact
        );
        assert_eq!(
            first.autoplay_benchmark_config().label.name,
            "fast_benchmark"
        );
        assert_eq!(first.root_benchmark_config().label.name, "fast_benchmark");
    }

    #[test]
    fn preset_comparison_is_reproducible_and_stably_ranked() {
        let suite = BenchmarkSuite::from_base_seed("preset-compare", 910, 1);
        let mut fast = fast_benchmark();
        fast.autoplay.max_steps = 0;
        let mut balanced = balanced_benchmark();
        balanced.autoplay.max_steps = 0;

        let first = compare_experiment_presets_on_suite(
            &suite,
            &[fast.clone(), balanced.clone()],
            PresetRankingMetric::Efficiency,
        )
        .unwrap();
        let second = compare_experiment_presets_on_suite(
            &suite,
            &[fast, balanced],
            PresetRankingMetric::Efficiency,
        )
        .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.entries.len(), 2);
        assert_eq!(first.entries[0].preset_name, "balanced_benchmark");
        assert_eq!(first.entries[0].efficiency_rank, 1);
        assert_eq!(first.entries[1].efficiency_rank, 2);
    }

    #[test]
    fn preset_comparison_export_shape_is_stable() {
        let suite = BenchmarkSuite::from_base_seed("preset-export", 920, 1);
        let mut fast = fast_benchmark();
        fast.autoplay.max_steps = 0;
        let summary =
            compare_experiment_presets_on_suite(&suite, &[fast], PresetRankingMetric::WinRate)
                .unwrap();

        let json = summary.to_json_summary().unwrap();
        assert!(json.contains("\"ranking_metric\""));
        assert!(json.contains("fast_benchmark"));

        let csv = summary.to_csv_summary();
        assert!(csv.starts_with("ranking_metric,preset_name,backend"));
        assert!(csv.contains("fast_benchmark"));
    }

    #[test]
    fn pimc_does_not_mutate_original_belief_state() {
        let deal = ExperimentRunner.generate_deal(DealSeed(8)).unwrap();
        let before = deal.belief_state.clone();

        recommend_move_pimc(
            &deal.belief_state,
            DeterministicSearchConfig::default(),
            fast_pimc_config(9),
        )
        .unwrap();

        assert_eq!(deal.belief_state, before);
    }

    #[test]
    fn sampled_full_worlds_remain_valid_against_originating_belief() {
        let deal = ExperimentRunner.generate_deal(DealSeed(11)).unwrap();
        let batch = PimcWorldBatch::sample(&deal.belief_state, 3, DealSeed(12)).unwrap();

        for sample in batch.samples {
            validate_sample_against_belief(&sample.full_state, &deal.belief_state).unwrap();
        }
    }

    #[test]
    fn sampler_assigns_each_unseen_card_once_without_weighting_surface() {
        let deal = ExperimentRunner.generate_deal(DealSeed(13)).unwrap();
        let batch = PimcWorldBatch::sample(&deal.belief_state, 1, DealSeed(14)).unwrap();
        let sample = &batch.samples[0].full_state;
        let sampled_cards = UnseenCardSet::from_cards(
            sample
                .hidden_assignments
                .iter()
                .map(|assignment| assignment.card),
        )
        .unwrap();

        assert_eq!(sampled_cards, deal.belief_state.unseen_cards);
    }

    #[test]
    fn autoplay_does_not_mutate_original_full_state() {
        let full = forced_reveal_full_state();
        let before = full.clone();

        let result = play_game_with_planner(
            &full,
            &fast_solver_config(),
            &fast_autoplay_config(PlannerBackend::BeliefUctLateExact),
        )
        .unwrap();

        assert_eq!(full, before);
        assert!(!result.trace.is_empty());
    }

    #[test]
    fn autoplay_belief_tracks_true_reveals() {
        let full = forced_reveal_full_state();
        let result = play_game_with_planner(
            &full,
            &fast_solver_config(),
            &fast_autoplay_config(PlannerBackend::BeliefUctLateExact),
        )
        .unwrap();

        let reveal_step = result
            .trace
            .steps
            .iter()
            .find(|step| step.revealed_card.is_some())
            .expect("forced state should reveal a hidden card");

        assert_eq!(reveal_step.revealed_card, Some("Ks".parse().unwrap()));
        assert_eq!(reveal_step.hidden_count_before, 1);
        assert_eq!(reveal_step.hidden_count_after, 0);
        validate_belief_against_full_state(&result.final_belief, &result.final_full_state).unwrap();
    }

    #[test]
    fn autoplay_detects_win_and_no_move_terminations() {
        let won = play_game_with_planner(
            &won_full_state(),
            &fast_solver_config(),
            &fast_autoplay_config(PlannerBackend::BeliefUctLateExact),
        )
        .unwrap();
        assert!(won.won);
        assert_eq!(won.termination, AutoplayTermination::Win);
        assert_eq!(won.trace.len(), 0);

        let lost = play_game_with_planner(
            &no_move_full_state(),
            &fast_solver_config(),
            &fast_autoplay_config(PlannerBackend::BeliefUctLateExact),
        )
        .unwrap();
        assert!(!lost.won);
        assert_eq!(lost.termination, AutoplayTermination::NoLegalMove);
    }

    #[test]
    fn same_seed_and_config_give_reproducible_autoplay_records() {
        let runner = ExperimentRunner;
        let suite = BenchmarkSuite::from_seed_range("autoplay-repro", 500, 2);
        let mut autoplay = fast_autoplay_config(PlannerBackend::BeliefUct);
        autoplay.max_steps = 0;
        let config = AutoplayBenchmarkConfig {
            label: EngineConfigLabel::new("autoplay"),
            solver: fast_solver_config(),
            autoplay,
        };

        let first = runner.run_autoplay_benchmark(&suite, &config).unwrap();
        let second = runner.run_autoplay_benchmark(&suite, &config).unwrap();

        assert_eq!(first.records, second.records);
        assert_eq!(first.win_rate, second.win_rate);
    }

    #[test]
    fn autoplay_ab_comparison_uses_identical_deal_suite() {
        let runner = ExperimentRunner;
        let suite = BenchmarkSuite::from_seed_range("autoplay-ab", 600, 2);
        let mut baseline_autoplay = fast_autoplay_config(PlannerBackend::BeliefUct);
        baseline_autoplay.max_steps = 0;
        let mut candidate_autoplay = fast_autoplay_config(PlannerBackend::BeliefUctLateExact);
        candidate_autoplay.max_steps = 0;
        let baseline = AutoplayBenchmarkConfig {
            label: EngineConfigLabel::new("base"),
            solver: fast_solver_config(),
            autoplay: baseline_autoplay,
        };
        let candidate = AutoplayBenchmarkConfig {
            label: EngineConfigLabel::new("cand"),
            solver: fast_solver_config(),
            autoplay: candidate_autoplay,
        };

        let result = runner
            .run_autoplay_paired_comparison(&suite, &baseline, &candidate)
            .unwrap();

        let baseline_seeds = result
            .baseline
            .records
            .iter()
            .map(|record| record.seed)
            .collect::<Vec<_>>();
        let candidate_seeds = result
            .candidate
            .records
            .iter()
            .map(|record| record.seed)
            .collect::<Vec<_>>();

        assert_eq!(baseline_seeds, suite.seeds);
        assert_eq!(candidate_seeds, suite.seeds);
        assert_eq!(
            result
                .paired_records
                .iter()
                .map(|record| record.seed)
                .collect::<Vec<_>>(),
            suite.seeds
        );
    }

    #[test]
    fn full_game_benchmark_aggregates_metrics() {
        let runner = ExperimentRunner;
        let suite = BenchmarkSuite::from_seed_range("autoplay-metrics", 700, 3);
        let config = fast_autoplay_benchmark_config("metrics", PlannerBackend::BeliefUct);

        let result = runner.run_autoplay_benchmark(&suite, &config).unwrap();

        assert_eq!(result.games, 3);
        assert_eq!(result.records.len(), 3);
        assert_eq!(result.wins + result.losses, 3);
        assert_eq!(result.average_moves_per_game, 0.0);
        assert_eq!(
            result
                .terminations
                .iter()
                .find(|entry| entry.termination == AutoplayTermination::StepLimit)
                .map(|entry| entry.count),
            Some(3)
        );
    }

    #[test]
    fn autoplay_json_and_csv_exports_have_stable_shape() {
        let runner = ExperimentRunner;
        let suite = BenchmarkSuite::from_base_seed("export", 800, 1);
        let config = fast_autoplay_benchmark_config("export-config", PlannerBackend::BeliefUct);
        let result = runner.run_autoplay_benchmark(&suite, &config).unwrap();

        let json = result.to_json_summary().unwrap();
        let report: AutoplayBenchmarkSummaryReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report.config_preset_name, "export-config");
        assert_eq!(report.backend, PlannerBackend::BeliefUct);
        assert_eq!(report.suite.base_seed, Some(DealSeed(800)));

        let csv = result.to_csv_summary();
        assert!(csv.starts_with("benchmark_kind,backend,config_preset_name"));
        assert!(csv.contains("BeliefUct"));

        let game_csv = result.to_game_csv();
        assert!(game_csv.starts_with("backend,config_preset_name,suite_name,seed"));
        assert!(game_csv.contains("800"));
    }

    #[test]
    fn paired_autoplay_summary_counts_same_outcomes_and_ci_fields() {
        let runner = ExperimentRunner;
        let suite = BenchmarkSuite::from_base_seed("paired-summary", 850, 2);
        let baseline = fast_autoplay_benchmark_config("A", PlannerBackend::BeliefUct);
        let candidate = fast_autoplay_benchmark_config("B", PlannerBackend::BeliefUctLateExact);

        let result = runner
            .run_autoplay_paired_comparison(&suite, &baseline, &candidate)
            .unwrap();

        assert_eq!(result.baseline_wins, result.baseline.wins);
        assert_eq!(result.candidate_wins, result.candidate.wins);
        assert_eq!(result.same_outcome_count, result.paired_records.len());
        assert_eq!(result.paired_standard_error, 0.0);
        assert_eq!(result.ci_lower, result.paired_win_rate_delta);
        assert_eq!(result.ci_upper, result.paired_win_rate_delta);

        let csv = result.to_csv_summary();
        assert!(csv.contains("wins_a,wins_b,paired_win_difference"));
        let json = result.to_json_summary().unwrap();
        assert!(json.contains("same_outcome_count"));
    }

    #[test]
    fn benchmark_export_does_not_change_computed_results() {
        let runner = ExperimentRunner;
        let suite = BenchmarkSuite::from_base_seed("no-mutate-export", 875, 2);
        let config = fast_autoplay_benchmark_config("stable", PlannerBackend::BeliefUct);
        let result = runner.run_autoplay_benchmark(&suite, &config).unwrap();
        let before = result.clone();

        let _ = export_autoplay_benchmark_json(&result).unwrap();
        let _ = export_autoplay_benchmark_csv(&result);
        let _ = export_autoplay_game_csv(&result);

        assert_eq!(result, before);
    }

    #[test]
    fn autoplay_can_use_pimc_backend() {
        let full = forced_reveal_full_state();
        let result = play_game_with_planner(
            &full,
            &fast_solver_config(),
            &fast_autoplay_config(PlannerBackend::Pimc),
        )
        .unwrap();

        assert!(!result.trace.is_empty());
        assert!(matches!(
            result.trace.steps[0].chosen_move.kind,
            MacroMoveKind::MoveTopToFoundation { .. }
        ));
    }
}
