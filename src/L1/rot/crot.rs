//! `crot` — plane rotation with a real cosine and a complex sine:
//! (x, y) ← (c·x + s·y, c·y − conj(s)·x), reference BLAS's `crot`.
//!
//! Implementation: the scalar complex definition per element, in the
//! canonical product order (`c32.rs`); the four products of each
//! element are independent, so there is no fold and the result is
//! bit-for-bit the definition on every target. Not streamed through
//! the lanes: a rotation is applied to a pair of vectors once, and the
//! real-sine form (`csrot`) is the one the roofline measures.

use crate::c32::C32;
use crate::Error;

/// Applies the plane rotation `(x, y) ← (c·x + s·y, c·y − conj(s)·x)`
/// over two slices of equal length, `c` real and `s` complex.
///
/// Errors: `x` and `y` differ in length (argument 2).
pub fn crot(x: &mut [C32], y: &mut [C32], c: f32, s: C32) -> Result<(), Error> {
	crate::error::same_len("crot", 2, x.len(), y.len())?;
	crot_k(x, y, c, s);
	Ok(())
}

/// Apply the rotation. Panics on length mismatch.
pub(crate) fn crot_k(x: &mut [C32], y: &mut [C32], c: f32, s: C32) {
	assert_eq!(x.len(), y.len(), "crot: length mismatch");
	let sc = s.conj();
	for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
		let (xv, yv) = (*xi, *yi);
		*xi = xv.scale(c) + s * yv;
		*yi = yv.scale(c) - sc * xv;
	}
}
