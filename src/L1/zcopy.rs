//! `zcopy` — complex vector copy: y ← x.
//!
//! Implementation: pure delegation — copying interleaved complex IS
//! `dcopy` on the 2n-real view (bit moves have no complex structure).
//! The tuned d-stream carries over for free.

use crate::c64::{as_re, as_re_mut, C64};
use crate::Error;
use crate::L1::dcopy_k;

/// `y ← x` over two slices of equal length.
///
/// Errors: `x` and `y` differ in length (argument 2).
pub fn zcopy(x: &[C64], y: &mut [C64]) -> Result<(), Error> {
	crate::error::same_len("zcopy", 2, x.len(), y.len())?;
	zcopy_k(x, y);
	Ok(())
}

/// y ← x. Panics on length mismatch.
pub(crate) fn zcopy_k(x: &[C64], y: &mut [C64]) {
	dcopy_k(as_re(x), as_re_mut(y));
}
