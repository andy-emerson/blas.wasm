//! `sger` — outer-product (rank-1) update: A ← A + αxyᵀ.
//!
//! Implementation: column-saxpy — one Level 1 `saxpy` stream per column
//! (column j gets α·`y[j]` times x).

use super::check_mat;
use crate::{Error};
use crate::L1::saxpy_k;

/// `A ← αxyᵀ + A`. `A` is `m×n` at leading dimension `lda`; `x` has
/// `m` elements and `y` has `n`. Longer slices are used up to those
/// lengths.
///
/// Errors: `lda < max(1, m)` (argument 7); `x`, `y` or `a` shorter than
/// its extent (arguments 4, 5, 6).
pub fn sger(m: usize, n: usize, alpha: f32, x: &[f32], y: &[f32], a: &mut [f32], lda: usize) -> Result<(), Error> {
	const R: &str = "sger";
	crate::error::vec(R, 4, x.len(), m)?;
	crate::error::vec(R, 5, y.len(), n)?;
	crate::error::mat(R, 6, 7, a.len(), m, n, lda)?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	sger_k(alpha, m, n, a, lda, &x[..m], &y[..n]);
	Ok(())
}

/// A ← A + αxyᵀ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub(crate) fn sger_k(alpha: f32, nrows: usize, ncols: usize, a: &mut [f32], cs: usize, x: &[f32], y: &[f32]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "sger: x length mismatch");
	assert_eq!(y.len(), ncols, "sger: y length mismatch");
	for j in 0..ncols {
		saxpy_k(alpha * y[j], x, &mut a[j * cs..j * cs + nrows]);
	}
}
