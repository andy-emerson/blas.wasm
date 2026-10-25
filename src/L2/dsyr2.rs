//! `dsyr2` — symmetric rank-2 update: A ← A + α(xyᵀ + yxᵀ), one
//! triangle stored.
//!
//! Implementation: column-daxpy — two Level 1 `daxpy` streams per stored
//! column segment (the second rides the cache the first just warmed).

use super::check_mat;
use crate::{Error, Uplo};
use crate::L1::daxpy_k;

/// `A ← αxyᵀ + αyxᵀ + A` for a symmetric `A` of order
/// `n`, the triangle named by `uplo` stored, at leading dimension
/// `lda`; `x` and `y` have `n` elements.
///
/// Errors: `lda < max(1, n)` (argument 7); `x`, `y` or `a` shorter than
/// its extent (arguments 4, 5, 6).
pub fn dsyr2(uplo: Uplo, n: usize, alpha: f64, x: &[f64], y: &[f64], a: &mut [f64], lda: usize) -> Result<(), Error> {
	const R: &str = "dsyr2";
	crate::error::vec(R, 4, x.len(), n)?;
	crate::error::vec(R, 5, y.len(), n)?;
	crate::error::mat(R, 6, 7, a.len(), n, n, lda)?;
	if n == 0 {
		return Ok(());
	}
	dsyr2_k(alpha, n, a, lda, uplo.is_upper(), &x[..n], &y[..n]);
	Ok(())
}

/// A ← A + α(xyᵀ + yxᵀ), A symmetric n×n at column stride `cs`,
/// `upper` (or lower) triangle stored.
pub(crate) fn dsyr2_k(alpha: f64, n: usize, a: &mut [f64], cs: usize, upper: bool, x: &[f64], y: &[f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "dsyr2: x length mismatch");
	assert_eq!(y.len(), n, "dsyr2: y length mismatch");
	for j in 0..n {
		let (ty, tx) = (alpha * y[j], alpha * x[j]);
		if upper {
			let col = &mut a[j * cs..j * cs + j + 1];
			daxpy_k(ty, &x[..=j], col);
			daxpy_k(tx, &y[..=j], col);
		} else {
			let col = &mut a[j * cs + j..j * cs + n];
			daxpy_k(ty, &x[j..], col);
			daxpy_k(tx, &y[j..], col);
		}
	}
}
