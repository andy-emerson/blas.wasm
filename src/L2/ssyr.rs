//! `ssyr` — symmetric rank-1 update: A ← A + αxxᵀ, one triangle stored.
//!
//! Implementation: column-saxpy — one Level 1 `saxpy` stream per stored
//! column segment.

use super::check_mat;
use crate::{Error, Uplo};
use crate::L1::saxpy_k;

/// `A ← αxxᵀ + A` for a symmetric `A` of order `n`, the triangle named
/// by `uplo` stored, at leading dimension `lda`; `x` has `n` elements.
///
/// Errors: `lda < max(1, n)` (argument 6); `x` or `a` shorter than its
/// extent (arguments 4, 5).
pub fn ssyr(uplo: Uplo, n: usize, alpha: f32, x: &[f32], a: &mut [f32], lda: usize) -> Result<(), Error> {
	const R: &str = "ssyr";
	crate::error::vec(R, 4, x.len(), n)?;
	crate::error::mat(R, 5, 6, a.len(), n, n, lda)?;
	if n == 0 {
		return Ok(());
	}
	ssyr_k(alpha, n, a, lda, uplo.is_upper(), &x[..n]);
	Ok(())
}

/// A ← A + αxxᵀ, A symmetric n×n at column stride `cs`, `upper` (or
/// lower) triangle stored.
pub(crate) fn ssyr_k(alpha: f32, n: usize, a: &mut [f32], cs: usize, upper: bool, x: &[f32]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "ssyr: x length mismatch");
	for j in 0..n {
		let t = alpha * x[j];
		if upper {
			saxpy_k(t, &x[..=j], &mut a[j * cs..j * cs + j + 1]);
		} else {
			saxpy_k(t, &x[j..], &mut a[j * cs + j..j * cs + n]);
		}
	}
}
