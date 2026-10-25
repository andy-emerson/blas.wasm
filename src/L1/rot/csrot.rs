//! `csrot` — apply a REAL plane rotation to complex f32 vectors.
//! Pure delegation: a real rotation acts on re and im independently,
//! so it IS `srot` on the 2n-real views (see `zdrot`).

use crate::c32::{as_re_mut, C32};
use crate::Error;
use crate::L1::srot_k;

/// Applies the plane rotation `(x, y) ← (c·x + s·y, c·y − s·x)` over
/// two slices of equal length.
///
/// Errors: `x` and `y` differ in length (argument 2).
pub fn csrot(x: &mut [C32], y: &mut [C32], c: f32, s: f32) -> Result<(), Error> {
	crate::error::same_len("csrot", 2, x.len(), y.len())?;
	csrot_k(x, y, c, s);
	Ok(())
}

/// Apply the rotation with real cosine `c` and real sine `s`.
/// Panics on length mismatch.
pub(crate) fn csrot_k(x: &mut [C32], y: &mut [C32], c: f32, s: f32) {
	srot_k(as_re_mut(x), as_re_mut(y), c, s);
}
