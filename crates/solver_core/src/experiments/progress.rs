//! Lightweight progress events for long experiment and benchmark runs.
//!
//! Progress reporting is deliberately callback-based and optional. Experiment
//! runners emit coarse per-game events; callers decide whether to print them,
//! store them, or ignore them.

use serde::{Deserialize, Serialize};

use super::PlannerBackend;

/// Long-running experiment command category.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgressCommandKind {
    /// Single full-game autoplay benchmark.
    Autoplay,
    /// Paired A/B autoplay comparison.
    Compare,
    /// Repeated paired autoplay comparison.
    RepeatedCompare,
    /// Multi-preset autoplay comparison.
    ComparePresets,
    /// Dataset export over autoplay traces.
    DatasetExport,
}

/// Coarse progress snapshot emitted by benchmark/autoplay loops.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgressEvent {
    /// Command category currently running.
    pub command: ProgressCommandKind,
    /// Preset/config currently being evaluated, when applicable.
    pub preset_name: Option<String>,
    /// Planner backend used by the current preset, when applicable.
    pub backend: Option<PlannerBackend>,
    /// One-based preset index for multi-preset comparisons.
    pub preset_index: Option<usize>,
    /// Total preset count for multi-preset comparisons.
    pub preset_total: Option<usize>,
    /// One-based repetition index for repeated comparisons.
    pub repetition_index: Option<usize>,
    /// Total repetition count for repeated comparisons.
    pub repetition_total: Option<usize>,
    /// Completed games in the current suite/config.
    pub game_index: usize,
    /// Total games in the current suite/config.
    pub game_total: usize,
    /// Elapsed milliseconds for the current suite/config segment.
    pub elapsed_ms: u64,
    /// Estimated milliseconds remaining for the current suite/config segment.
    pub eta_ms: Option<u64>,
    /// Partial wins observed in the current suite/config segment.
    pub wins: usize,
    /// Partial losses observed in the current suite/config segment.
    pub losses: usize,
}

/// Sink for optional experiment progress events.
pub trait ProgressReporter {
    /// Receives one coarse progress event.
    fn report(&mut self, event: &ProgressEvent);
}

/// Progress sink that discards every event.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub struct NoopProgressReporter;

impl ProgressReporter for NoopProgressReporter {
    fn report(&mut self, _event: &ProgressEvent) {}
}

impl<F> ProgressReporter for F
where
    F: FnMut(&ProgressEvent),
{
    fn report(&mut self, event: &ProgressEvent) {
        self(event);
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) struct ProgressContext {
    pub(crate) command: ProgressCommandKind,
    pub(crate) preset_index: Option<usize>,
    pub(crate) preset_total: Option<usize>,
    pub(crate) repetition_index: Option<usize>,
    pub(crate) repetition_total: Option<usize>,
}

impl ProgressContext {
    pub(crate) const fn new(command: ProgressCommandKind) -> Self {
        Self {
            command,
            preset_index: None,
            preset_total: None,
            repetition_index: None,
            repetition_total: None,
        }
    }

    pub(crate) const fn with_preset(mut self, index: usize, total: usize) -> Self {
        self.preset_index = Some(index);
        self.preset_total = Some(total);
        self
    }

    pub(crate) const fn with_repetition(mut self, index: usize, total: usize) -> Self {
        self.repetition_index = Some(index);
        self.repetition_total = Some(total);
        self
    }
}
