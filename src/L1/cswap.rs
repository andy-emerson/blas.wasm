//! `cswap` — exchange two complex f32 vectors. Pure delegation:
//! `sswap` on the 2n-real views.

use crate::c32::{as_re_mut, C32};
use crate::Error;
use crate::L1::sswap_k;

/// `x ↔ y` over two slices of equal length.
///
/// Errors: `x` and `y` differ in length (argument 2).
pub fn cswap(x: &mut [C32], y: &mut [C32]) -> Result<(), Error> {
	crate::error::same_len("cswap", 2, x.len(), y.len())?;
	cswap_k(x, y);
	Ok(())
}

/// x ↔ y. Panics on length mismatch.
pub(crate) fn cswap_k(x: &mut [C32], y: &mut [C32]) {
	sswap_k(as_re_mut(x), as_re_mut(y));
}
