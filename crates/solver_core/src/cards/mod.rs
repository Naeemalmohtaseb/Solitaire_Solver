//! Compact card encoding and card property utilities.

use serde::{Deserialize, Serialize};

use crate::error::{SolverError, SolverResult};

/// Compact card identifier using suit-major indexing in the range 0..=51.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Card(u8);

impl Card {
    /// Total number of cards in a standard deck.
    pub const COUNT: usize = 52;

    /// Creates a checked card from its compact index.
    pub fn new(index: u8) -> SolverResult<Self> {
        if usize::from(index) < Self::COUNT {
            Ok(Self(index))
        } else {
            Err(SolverError::InvalidCardIndex(u16::from(index)))
        }
    }

    /// Creates a card from a suit and rank.
    pub const fn from_suit_rank(suit: Suit, rank: Rank) -> Self {
        Self(suit.offset() * 13 + rank.offset())
    }

    /// Returns the compact 0..=51 card index.
    pub const fn index(self) -> u8 {
        self.0
    }

    /// Returns the card suit.
    pub const fn suit(self) -> Suit {
        match self.0 / 13 {
            0 => Suit::Clubs,
            1 => Suit::Diamonds,
            2 => Suit::Hearts,
            _ => Suit::Spades,
        }
    }

    /// Returns the card rank.
    pub const fn rank(self) -> Rank {
        match self.0 % 13 {
            0 => Rank::Ace,
            1 => Rank::Two,
            2 => Rank::Three,
            3 => Rank::Four,
            4 => Rank::Five,
            5 => Rank::Six,
            6 => Rank::Seven,
            7 => Rank::Eight,
            8 => Rank::Nine,
            9 => Rank::Ten,
            10 => Rank::Jack,
            11 => Rank::Queen,
            _ => Rank::King,
        }
    }

    /// Returns the card color.
    pub const fn color(self) -> Color {
        self.suit().color()
    }
}

/// Card suit.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Suit {
    /// Clubs, encoded first.
    Clubs,
    /// Diamonds, encoded second.
    Diamonds,
    /// Hearts, encoded third.
    Hearts,
    /// Spades, encoded fourth.
    Spades,
}

impl Suit {
    /// Returns the suit's compact 0..=3 offset.
    pub const fn offset(self) -> u8 {
        match self {
            Self::Clubs => 0,
            Self::Diamonds => 1,
            Self::Hearts => 2,
            Self::Spades => 3,
        }
    }

    /// Returns the suit color.
    pub const fn color(self) -> Color {
        match self {
            Self::Clubs | Self::Spades => Color::Black,
            Self::Diamonds | Self::Hearts => Color::Red,
        }
    }
}

/// Card rank.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Rank {
    /// Ace.
    Ace,
    /// Two.
    Two,
    /// Three.
    Three,
    /// Four.
    Four,
    /// Five.
    Five,
    /// Six.
    Six,
    /// Seven.
    Seven,
    /// Eight.
    Eight,
    /// Nine.
    Nine,
    /// Ten.
    Ten,
    /// Jack.
    Jack,
    /// Queen.
    Queen,
    /// King.
    King,
}

impl Rank {
    /// Returns the 0..=12 rank offset used in compact card ids.
    pub const fn offset(self) -> u8 {
        match self {
            Self::Ace => 0,
            Self::Two => 1,
            Self::Three => 2,
            Self::Four => 3,
            Self::Five => 4,
            Self::Six => 5,
            Self::Seven => 6,
            Self::Eight => 7,
            Self::Nine => 8,
            Self::Ten => 9,
            Self::Jack => 10,
            Self::Queen => 11,
            Self::King => 12,
        }
    }

    /// Returns the traditional rank value Ace=1 through King=13.
    pub const fn value(self) -> u8 {
        self.offset() + 1
    }
}

/// Card color.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Color {
    /// Black suit color.
    Black,
    /// Red suit color.
    Red,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn card_indices_cover_standard_deck() {
        let first = Card::new(0).expect("card 0 should be valid");
        let last = Card::new(51).expect("card 51 should be valid");

        assert_eq!(first.suit(), Suit::Clubs);
        assert_eq!(first.rank(), Rank::Ace);
        assert_eq!(last.suit(), Suit::Spades);
        assert_eq!(last.rank(), Rank::King);
        assert!(Card::new(52).is_err());
    }

    #[test]
    fn suit_rank_constructor_matches_encoding() {
        let queen_hearts = Card::from_suit_rank(Suit::Hearts, Rank::Queen);

        assert_eq!(queen_hearts.index(), 37);
        assert_eq!(queen_hearts.color(), Color::Red);
    }
}
