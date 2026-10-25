//! `ccopy` — complex f32 vector copy: y ← x. Pure delegation:
//! `scopy` on the 2n-real view.

use crate::c32::{as_re, as_re_mut, C32};
use crate::Error;
use crate::L1::scopy_k;

/// `y ← x` over two slices of equal length.
///
/// Errors: `x` and `y` differ in length (argument 2).
pub fn ccopy(x: &[C32], y: &mut [C32]) -> Result<(), Error> {
	crate::error::same_len("ccopy", 2, x.len(), y.len())?;
	ccopy_k(x, y);
	Ok(())
}

/// y ← x. Panics on length mismatch.
pub(crate) fn ccopy_k(x: &[C32], y: &mut [C32]) {
	scopy_k(as_re(x), as_re_mut(y));
}
