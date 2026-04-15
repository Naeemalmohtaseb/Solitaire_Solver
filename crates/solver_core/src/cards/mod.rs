//! Compact card encoding and card property utilities.

use std::{fmt, str::FromStr};

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

    /// Returns true if this card is red.
    pub const fn is_red(self) -> bool {
        matches!(self.color(), Color::Red)
    }

    /// Returns true if this card is black.
    pub const fn is_black(self) -> bool {
        matches!(self.color(), Color::Black)
    }

    /// Returns true if `self` can be placed onto `other` in the tableau.
    ///
    /// This is a domain-level card relation only; it does not generate moves.
    pub const fn can_tableau_stack_on(self, other: Card) -> bool {
        self.color().is_opposite(other.color()) && self.rank().value() + 1 == other.rank().value()
    }

    /// Returns true if this card can be advanced to a same-suit foundation whose
    /// current top rank is `foundation_top`.
    ///
    /// The caller is responsible for passing the top rank for this card's suit.
    pub const fn can_move_to_foundation(self, foundation_top: Option<Rank>) -> bool {
        match foundation_top {
            None => matches!(self.rank(), Rank::Ace),
            Some(top) => self.rank().value() == top.value() + 1,
        }
    }
}

impl fmt::Display for Card {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.rank(), self.suit())
    }
}

impl FromStr for Card {
    type Err = SolverError;

    fn from_str(text: &str) -> SolverResult<Self> {
        let mut chars = text.chars();
        let rank_char = chars
            .next()
            .ok_or_else(|| SolverError::InvalidCardText(text.to_string()))?;
        let suit_char = chars
            .next()
            .ok_or_else(|| SolverError::InvalidCardText(text.to_string()))?;

        if chars.next().is_some() {
            return Err(SolverError::InvalidCardText(text.to_string()));
        }

        Ok(Self::from_suit_rank(
            Suit::from_char(suit_char)?,
            Rank::from_char(rank_char)?,
        ))
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
    /// Creates a suit from its compact 0..=3 offset.
    pub fn from_index(index: u8) -> SolverResult<Self> {
        match index {
            0 => Ok(Self::Clubs),
            1 => Ok(Self::Diamonds),
            2 => Ok(Self::Hearts),
            3 => Ok(Self::Spades),
            _ => Err(SolverError::InvalidSuitIndex(index)),
        }
    }

    /// Creates a suit from a display character.
    pub fn from_char(ch: char) -> SolverResult<Self> {
        match ch.to_ascii_lowercase() {
            'c' => Ok(Self::Clubs),
            'd' => Ok(Self::Diamonds),
            'h' => Ok(Self::Hearts),
            's' => Ok(Self::Spades),
            _ => Err(SolverError::InvalidCardText(ch.to_string())),
        }
    }

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

impl fmt::Display for Suit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ch = match self {
            Self::Clubs => 'c',
            Self::Diamonds => 'd',
            Self::Hearts => 'h',
            Self::Spades => 's',
        };
        write!(f, "{ch}")
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
    /// Creates a rank from the traditional Ace=1 through King=13 value.
    pub fn from_value(value: u8) -> SolverResult<Self> {
        match value {
            1 => Ok(Self::Ace),
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            5 => Ok(Self::Five),
            6 => Ok(Self::Six),
            7 => Ok(Self::Seven),
            8 => Ok(Self::Eight),
            9 => Ok(Self::Nine),
            10 => Ok(Self::Ten),
            11 => Ok(Self::Jack),
            12 => Ok(Self::Queen),
            13 => Ok(Self::King),
            _ => Err(SolverError::InvalidRankValue(value)),
        }
    }

    /// Creates a rank from a display character.
    pub fn from_char(ch: char) -> SolverResult<Self> {
        match ch.to_ascii_uppercase() {
            'A' => Ok(Self::Ace),
            '2' => Ok(Self::Two),
            '3' => Ok(Self::Three),
            '4' => Ok(Self::Four),
            '5' => Ok(Self::Five),
            '6' => Ok(Self::Six),
            '7' => Ok(Self::Seven),
            '8' => Ok(Self::Eight),
            '9' => Ok(Self::Nine),
            'T' => Ok(Self::Ten),
            'J' => Ok(Self::Jack),
            'Q' => Ok(Self::Queen),
            'K' => Ok(Self::King),
            _ => Err(SolverError::InvalidCardText(ch.to_string())),
        }
    }

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

    /// Returns the next higher rank, if any.
    pub fn successor(self) -> Option<Self> {
        Self::from_value(self.value() + 1).ok()
    }

    /// Returns the next lower rank, if any.
    pub fn predecessor(self) -> Option<Self> {
        self.value()
            .checked_sub(1)
            .and_then(|value| Self::from_value(value).ok())
    }
}

impl fmt::Display for Rank {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ch = match self {
            Self::Ace => 'A',
            Self::Two => '2',
            Self::Three => '3',
            Self::Four => '4',
            Self::Five => '5',
            Self::Six => '6',
            Self::Seven => '7',
            Self::Eight => '8',
            Self::Nine => '9',
            Self::Ten => 'T',
            Self::Jack => 'J',
            Self::Queen => 'Q',
            Self::King => 'K',
        };
        write!(f, "{ch}")
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

impl Color {
    /// Returns true when two card colors are opposite.
    pub const fn is_opposite(self, other: Self) -> bool {
        matches!(
            (self, other),
            (Self::Black, Self::Red) | (Self::Red, Self::Black)
        )
    }
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

    #[test]
    fn cards_parse_and_format_round_trip() {
        for text in ["Ah", "Td", "Ks", "2c", "QH"] {
            let card: Card = text.parse().expect("card text should parse");
            assert_eq!(
                card.to_string(),
                text.to_ascii_uppercase()[0..1].to_string() + &text[1..2].to_ascii_lowercase()
            );
        }

        assert!("10h".parse::<Card>().is_err());
        assert!("Xh".parse::<Card>().is_err());
    }

    #[test]
    fn card_property_helpers_match_klondike_relations() {
        let six_hearts: Card = "6h".parse().unwrap();
        let seven_spades: Card = "7s".parse().unwrap();
        let seven_diamonds: Card = "7d".parse().unwrap();
        let ace_clubs: Card = "Ac".parse().unwrap();
        let two_clubs: Card = "2c".parse().unwrap();

        assert!(six_hearts.is_red());
        assert!(seven_spades.is_black());
        assert!(six_hearts.can_tableau_stack_on(seven_spades));
        assert!(!six_hearts.can_tableau_stack_on(seven_diamonds));
        assert!(ace_clubs.can_move_to_foundation(None));
        assert!(two_clubs.can_move_to_foundation(Some(Rank::Ace)));
        assert!(!seven_spades.can_move_to_foundation(Some(Rank::Ace)));
    }
}
