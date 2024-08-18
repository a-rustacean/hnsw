#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct LCGRng {
    seed: u64,
}

impl LCGRng {
    pub fn new(seed: u64) -> Self {
        LCGRng { seed }
    }

    // Generate a random floating-point number between 0.0 and 1.0
    pub fn next_f64(&mut self) -> f64 {
        const MODULUS: u64 = 2_u64.pow(31) - 1;
        const MULTIPLIER: u64 = 16807;
        const INCREMENT: u64 = 0;

        self.seed = (MULTIPLIER.wrapping_mul(self.seed) + INCREMENT) % MODULUS;
        self.seed as f64 / MODULUS as f64
    }
}
