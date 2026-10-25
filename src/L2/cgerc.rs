//! `cgerc` — conjugated rank-1 update: A ← A + αx·yᴴ.
//!
//! Implementation: column-caxpy — identical to `cgeru` except column
//! j's scalar is α·conj(`y[j]`) (the conjugation lands on the scalar,
//! never on a stream).

use super::check_mat;
use crate::{Error};
use crate::c32::C32;
use crate::L1::caxpy_k;

/// `A ← αxyᴴ + A`. `A` is `m×n` at leading dimension `lda`; `x` has
/// `m` elements and `y` has `n`. Longer slices are used up to those
/// lengths.
///
/// Errors: `lda < max(1, m)` (argument 7); `x`, `y` or `a` shorter than
/// its extent (arguments 4, 5, 6).
pub fn cgerc(m: usize, n: usize, alpha: C32, x: &[C32], y: &[C32], a: &mut [C32], lda: usize) -> Result<(), Error> {
	const R: &str = "cgerc";
	crate::error::vec(R, 4, x.len(), m)?;
	crate::error::vec(R, 5, y.len(), n)?;
	crate::error::mat(R, 6, 7, a.len(), m, n, lda)?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	cgerc_k(alpha, m, n, a, lda, &x[..m], &y[..n]);
	Ok(())
}

/// A ← A + αx·yᴴ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub(crate) fn cgerc_k(alpha: C32, nrows: usize, ncols: usize, a: &mut [C32], cs: usize, x: &[C32], y: &[C32]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "cgerc: x length mismatch");
	assert_eq!(y.len(), ncols, "cgerc: y length mismatch");
	for j in 0..ncols {
		caxpy_k(alpha * y[j].conj(), x, &mut a[j * cs..j * cs + nrows]);
	}
}
