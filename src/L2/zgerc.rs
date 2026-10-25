//! `zgerc` — conjugated rank-1 update: A ← A + αx·yᴴ.
//!
//! Implementation: column-zaxpy — identical to `zgeru` except column
//! j's scalar is α·conj(`y[j]`) (the conjugation lands on the scalar,
//! never on a stream).

use super::check_mat;
use crate::{Error};
use crate::c64::C64;
use crate::L1::zaxpy_k;

/// `A ← αxyᴴ + A`. `A` is `m×n` at leading dimension `lda`; `x` has
/// `m` elements and `y` has `n`. Longer slices are used up to those
/// lengths.
///
/// Errors: `lda < max(1, m)` (argument 7); `x`, `y` or `a` shorter than
/// its extent (arguments 4, 5, 6).
pub fn zgerc(m: usize, n: usize, alpha: C64, x: &[C64], y: &[C64], a: &mut [C64], lda: usize) -> Result<(), Error> {
	const R: &str = "zgerc";
	crate::error::vec(R, 4, x.len(), m)?;
	crate::error::vec(R, 5, y.len(), n)?;
	crate::error::mat(R, 6, 7, a.len(), m, n, lda)?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	zgerc_k(alpha, m, n, a, lda, &x[..m], &y[..n]);
	Ok(())
}

/// A ← A + αx·yᴴ. A is nrows×ncols at column stride `cs`;
/// x has nrows elements, y has ncols.
pub(crate) fn zgerc_k(alpha: C64, nrows: usize, ncols: usize, a: &mut [C64], cs: usize, x: &[C64], y: &[C64]) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "zgerc: x length mismatch");
	assert_eq!(y.len(), ncols, "zgerc: y length mismatch");
	for j in 0..ncols {
		zaxpy_k(alpha * y[j].conj(), x, &mut a[j * cs..j * cs + nrows]);
	}
}
