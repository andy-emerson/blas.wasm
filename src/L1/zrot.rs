//! `zrot` — plane rotation with a real cosine and a complex sine:
//! (x, y) ← (c·x + s·y, c·y − conj(s)·x): LAPACK's auxiliary `zrot`
//! (reference BLAS has no complex-sine rotation; `zdrot` is its real one).
//!
//! Implementation: the scalar complex definition per element, in the
//! canonical product order (`c64.rs`); the four products of each
//! element are independent, so there is no fold and the result is
//! bit-for-bit the definition on every target. Not streamed through
//! the lanes: a rotation is applied to a pair of vectors once, and the
//! real-sine form (`zdrot`) is the one the roofline measures.

use crate::c64::C64;
use crate::Error;

/// Applies the plane rotation `(x, y) ← (c·x + s·y, c·y − conj(s)·x)`
/// over two slices of equal length, `c` real and `s` complex.
///
/// Errors: `x` and `y` differ in length (argument 2).
pub fn zrot(x: &mut [C64], y: &mut [C64], c: f64, s: C64) -> Result<(), Error> {
	crate::error::same_len("zrot", 2, x.len(), y.len())?;
	zrot_k(x, y, c, s);
	Ok(())
}

/// Apply the rotation. Panics on length mismatch.
pub(crate) fn zrot_k(x: &mut [C64], y: &mut [C64], c: f64, s: C64) {
	assert_eq!(x.len(), y.len(), "zrot: length mismatch");
	let sc = s.conj();
	for (xi, yi) in x.iter_mut().zip(y.iter_mut()) {
		let (xv, yv) = (*xi, *yi);
		*xi = xv.scale(c) + s * yv;
		*yi = yv.scale(c) - sc * xv;
	}
}
