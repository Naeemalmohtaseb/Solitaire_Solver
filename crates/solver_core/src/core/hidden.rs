//! Hidden tableau slot and assignment representation.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    cards::Card,
    error::{SolverError, SolverResult},
    types::ColumnId,
};

use super::state::VisibleState;

/// Stable identifier for one face-down tableau slot.
///
/// Slots are ordered by column, then by zero-based hidden depth. Depth `0` is
/// the deepest face-down card in a column; larger depths move upward toward the
/// reveal frontier.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HiddenSlot {
    /// Tableau column containing the hidden slot.
    pub column: ColumnId,
    /// Zero-based hidden depth within the column.
    pub depth: u8,
}

impl HiddenSlot {
    /// Creates a hidden slot identifier.
    pub const fn new(column: ColumnId, depth: u8) -> Self {
        Self { column, depth }
    }
}

impl fmt::Display for HiddenSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "C{}:H{}", self.column.index(), self.depth)
    }
}

/// Concrete card assignment for one hidden tableau slot.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenAssignment {
    /// Hidden slot being assigned.
    pub slot: HiddenSlot,
    /// Card occupying that slot.
    pub card: Card,
}

impl HiddenAssignment {
    /// Creates one hidden-slot assignment.
    pub const fn new(slot: HiddenSlot, card: Card) -> Self {
        Self { slot, card }
    }
}

/// Full mapping from hidden tableau slots to concrete cards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HiddenAssignments {
    /// Slot assignments in deterministic slot order.
    pub entries: Vec<HiddenAssignment>,
}

impl HiddenAssignments {
    /// Creates assignments and sorts them into deterministic slot order.
    pub fn new(mut entries: Vec<HiddenAssignment>) -> Self {
        entries.sort_by_key(|entry| entry.slot);
        Self { entries }
    }

    /// Returns an empty assignment mapping.
    pub fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Number of assigned hidden slots.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if there are no assignments.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterates assignments in deterministic slot order.
    pub fn iter(&self) -> impl Iterator<Item = &HiddenAssignment> {
        self.entries.iter()
    }

    /// Returns the assignment for a hidden slot.
    pub fn assignment_for_slot(&self, slot: HiddenSlot) -> Option<HiddenAssignment> {
        self.entries
            .binary_search_by_key(&slot, |entry| entry.slot)
            .ok()
            .map(|index| self.entries[index])
    }

    /// Returns the card assigned to a hidden slot.
    pub fn card_for_slot(&self, slot: HiddenSlot) -> Option<Card> {
        self.assignment_for_slot(slot)
            .map(|assignment| assignment.card)
    }

    /// Removes and returns the assignment for a hidden slot.
    pub fn remove_slot(&mut self, slot: HiddenSlot) -> SolverResult<HiddenAssignment> {
        let index = self
            .entries
            .binary_search_by_key(&slot, |entry| entry.slot)
            .map_err(|_| {
                SolverError::InvalidState(format!("missing hidden assignment for slot {slot}"))
            })?;
        Ok(self.entries.remove(index))
    }

    /// Inserts one assignment while preserving deterministic slot order.
    pub fn insert(&mut self, assignment: HiddenAssignment) -> SolverResult<()> {
        match self
            .entries
            .binary_search_by_key(&assignment.slot, |entry| entry.slot)
        {
            Ok(_) => Err(SolverError::DuplicateHiddenSlot(
                assignment.slot.to_string(),
            )),
            Err(index) => {
                self.entries.insert(index, assignment);
                self.validate_structure()
            }
        }
    }

    /// Validates local assignment structure.
    pub fn validate_structure(&self) -> SolverResult<()> {
        let mut card_mask = 0u64;
        let mut previous_slot: Option<HiddenSlot> = None;

        for entry in &self.entries {
            if let Some(previous) = previous_slot {
                if entry.slot < previous {
                    return Err(SolverError::InvalidState(
                        "hidden assignments must be stored in deterministic slot order".to_string(),
                    ));
                }
                if entry.slot == previous {
                    return Err(SolverError::DuplicateHiddenSlot(entry.slot.to_string()));
                }
            }
            previous_slot = Some(entry.slot);

            let bit = 1u64 << entry.card.index();
            if (card_mask & bit) != 0 {
                return Err(SolverError::DuplicateCard(entry.card));
            }
            card_mask |= bit;
        }

        Ok(())
    }

    /// Validates assignments against the visible tableau hidden-slot structure.
    pub fn validate_against_visible(&self, visible: &VisibleState) -> SolverResult<()> {
        self.validate_structure()?;

        let expected_slots = visible.hidden_slots();
        if self.entries.len() != expected_slots.len() {
            return Err(SolverError::HiddenAssignmentCountMismatch {
                expected: expected_slots.len(),
                actual: self.entries.len(),
            });
        }

        for (entry, expected_slot) in self.entries.iter().zip(expected_slots.iter()) {
            let hidden_count = visible.columns[usize::from(entry.slot.column.index())].hidden_count;
            if entry.slot.depth >= hidden_count {
                return Err(SolverError::HiddenSlotOutOfRange {
                    slot: entry.slot.to_string(),
                    hidden_count,
                });
            }
            if entry.slot != *expected_slot {
                return Err(SolverError::InvalidState(
                    "hidden assignments do not cover the visible hidden slots".to_string(),
                ));
            }
        }

        Ok(())
    }
}

impl Default for HiddenAssignments {
    fn default() -> Self {
        Self::empty()
    }
}

impl fmt::Display for HiddenAssignments {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[")?;
        for (index, entry) in self.entries.iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}={}", entry.slot, entry.card)?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_slots_order_by_column_then_depth() {
        let c0 = ColumnId::new(0).unwrap();
        let c1 = ColumnId::new(1).unwrap();
        let mut slots = [
            HiddenSlot::new(c1, 0),
            HiddenSlot::new(c0, 2),
            HiddenSlot::new(c0, 0),
        ];

        slots.sort();

        assert_eq!(
            slots,
            [
                HiddenSlot::new(c0, 0),
                HiddenSlot::new(c0, 2),
                HiddenSlot::new(c1, 0)
            ]
        );
        assert_eq!(HiddenSlot::new(c1, 0).to_string(), "C1:H0");
    }

    #[test]
    fn hidden_assignments_require_stable_slot_order() {
        let c0 = ColumnId::new(0).unwrap();
        let assignments = HiddenAssignments {
            entries: vec![
                HiddenAssignment::new(HiddenSlot::new(c0, 1), "Ac".parse().unwrap()),
                HiddenAssignment::new(HiddenSlot::new(c0, 0), "2c".parse().unwrap()),
            ],
        };

        assert!(assignments.validate_structure().is_err());
    }

    #[test]
    fn hidden_assignments_detect_adjacent_duplicate_slots() {
        let c0 = ColumnId::new(0).unwrap();
        let duplicate_slot = HiddenSlot::new(c0, 0);
        let assignments = HiddenAssignments::new(vec![
            HiddenAssignment::new(duplicate_slot, "Ac".parse().unwrap()),
            HiddenAssignment::new(duplicate_slot, "2c".parse().unwrap()),
        ]);

        assert!(matches!(
            assignments.validate_structure(),
            Err(SolverError::DuplicateHiddenSlot(slot)) if slot == duplicate_slot.to_string()
        ));
    }

    #[test]
    fn hidden_assignments_lookup_remove_and_insert_by_slot() {
        let c0 = ColumnId::new(0).unwrap();
        let slot0 = HiddenSlot::new(c0, 0);
        let slot1 = HiddenSlot::new(c0, 1);
        let mut assignments = HiddenAssignments::new(vec![
            HiddenAssignment::new(slot1, "2c".parse().unwrap()),
            HiddenAssignment::new(slot0, "Ac".parse().unwrap()),
        ]);

        assert_eq!(assignments.card_for_slot(slot0).unwrap().to_string(), "Ac");

        let removed = assignments.remove_slot(slot0).unwrap();
        assert_eq!(removed.slot, slot0);
        assert!(assignments.card_for_slot(slot0).is_none());

        assignments.insert(removed).unwrap();
        assert_eq!(assignments.entries[0].slot, slot0);
        assert_eq!(assignments.entries[1].slot, slot1);
    }
}
