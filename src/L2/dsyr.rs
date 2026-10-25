//! `dsyr` — symmetric rank-1 update: A ← A + αxxᵀ, one triangle stored.
//!
//! Implementation: column-daxpy — one Level 1 `daxpy` stream per stored
//! column segment.

use super::check_mat;
use crate::{Error, Uplo};
use crate::L1::daxpy_k;

/// `A ← αxxᵀ + A` for a symmetric `A` of order `n`, the triangle named
/// by `uplo` stored, at leading dimension `lda`; `x` has `n` elements.
///
/// Errors: `lda < max(1, n)` (argument 6); `x` or `a` shorter than its
/// extent (arguments 4, 5).
pub fn dsyr(uplo: Uplo, n: usize, alpha: f64, x: &[f64], a: &mut [f64], lda: usize) -> Result<(), Error> {
	const R: &str = "dsyr";
	crate::error::vec(R, 4, x.len(), n)?;
	crate::error::mat(R, 5, 6, a.len(), n, n, lda)?;
	dsyr_k(alpha, n, a, lda, uplo.is_upper(), &x[..n]);
	Ok(())
}

/// A ← A + αxxᵀ, A symmetric n×n at column stride `cs`, `upper` (or
/// lower) triangle stored.
pub(crate) fn dsyr_k(alpha: f64, n: usize, a: &mut [f64], cs: usize, upper: bool, x: &[f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "dsyr: x length mismatch");
	for j in 0..n {
		let t = alpha * x[j];
		if upper {
			daxpy_k(t, &x[..=j], &mut a[j * cs..j * cs + j + 1]);
		} else {
			daxpy_k(t, &x[j..], &mut a[j * cs + j..j * cs + n]);
		}
	}
}
