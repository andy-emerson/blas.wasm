//! `zswap` — exchange two complex vectors.
//!
//! Implementation: pure delegation — swapping interleaved complex IS
//! `dswap` on the 2n-real view. The tuned d-stream (SIMD swap raced
//! 1.2–1.3× over plain, campaign record step 2) carries over for free.

use crate::c64::{as_re_mut, C64};
use crate::Error;
use crate::L1::dswap_k;

/// `x ↔ y` over two slices of equal length.
///
/// Errors: `x` and `y` differ in length (argument 2).
pub fn zswap(x: &mut [C64], y: &mut [C64]) -> Result<(), Error> {
	crate::error::same_len("zswap", 2, x.len(), y.len())?;
	zswap_k(x, y);
	Ok(())
}

/// x ↔ y. Panics on length mismatch.
pub(crate) fn zswap_k(x: &mut [C64], y: &mut [C64]) {
	dswap_k(as_re_mut(x), as_re_mut(y));
}
