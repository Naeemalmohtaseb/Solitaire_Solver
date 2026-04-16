//! Small deterministic support utilities for the belief planner.

/// Tiny deterministic RNG used for planner sampling.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) struct PlannerRng {
    state: u64,
}

impl PlannerRng {
    pub(super) const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub(super) fn next_bounded(&mut self, upper_exclusive: usize) -> usize {
        debug_assert!(upper_exclusive > 0);
        (self.next_u64() % upper_exclusive as u64) as usize
    }

    pub(super) fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 / ((1u64 << 53) as f64)
    }
}

/// Stable structural hash helper used for cache identities and config fingerprints.
#[derive(Debug, Copy, Clone)]
pub(super) struct StableHash {
    value: u64,
}

impl StableHash {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

    pub(super) const fn new(seed: u64) -> Self {
        Self {
            value: 0xcbf2_9ce4_8422_2325 ^ seed,
        }
    }

    pub(super) const fn finish(self) -> u64 {
        self.value
    }

    pub(super) fn write_u8(&mut self, value: u8) {
        self.value ^= u64::from(value);
        self.value = self.value.wrapping_mul(Self::FNV_PRIME);
    }

    pub(super) fn write_bool(&mut self, value: bool) {
        self.write_u8(u8::from(value));
    }

    pub(super) fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }

    pub(super) fn write_u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.write_u8(byte);
        }
    }

    pub(super) fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    pub(super) fn write_f64(&mut self, value: f64) {
        self.write_u64(value.to_bits());
    }

    pub(super) fn write_str(&mut self, value: &str) {
        self.write_usize(value.len());
        for byte in value.as_bytes() {
            self.write_u8(*byte);
        }
    }
}
