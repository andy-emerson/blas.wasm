//! `zgeru` — unconjugated rank-1 update: A ← A + αxyᵀ.
//!
//! Implementation: column-zaxpy — one Level 1 `zaxpy` stream per
//! column (column j gets α·`y[j]` times x), exactly the `dger` shape.

use super::super::check_mat;
use crate::{Error};
use crate::c64::C64;
use crate::L1::zaxpy_k;

/// `A ← αxyᵀ + A`. `A` is `m×n` at leading dimension `lda`; `x` has
/// `m` elements and `y` has `n`. Longer slices are used up to those
/// lengths.
///
/// Errors: `lda < max(1, m)` (argument 7); `x`, `y` or `a` shorter than
/// its extent (arguments 4, 5, 6).
pub fn zgeru(m: usize, n: usize, alpha: C64, x: &[C64], y: &[C64], a: &mut [C64], lda: usize) -> Result<(), Error> {
	const R: &str = "zgeru";
	crate::error::vec(R, 4, x.len(), m)?;
	crate::error::vec(R, 5, y.len(), n)?;
	crate::error::mat(R, 6, 7, a.len(), m, n, lda)?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if alpha == C64::ZERO {
		return Ok(()); // reference: nothing is read or written
	}
	zgeru_k(alpha, m, n, a, lda, &x[..m], &y[..n]);
	Ok(())
}

/// A ← A + αxyᵀ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub(crate) fn zgeru_k(alpha: C64, nrows: usize, ncols: usize, a: &mut [C64], cs: usize, x: &[C64], y: &[C64]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "zgeru: x length mismatch");
	assert_eq!(y.len(), ncols, "zgeru: y length mismatch");
	for j in 0..ncols {
		zaxpy_k(alpha * y[j], x, &mut a[j * cs..j * cs + nrows]);
	}
}
