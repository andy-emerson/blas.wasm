//! `zher` — Hermitian rank-1 update: A ← A + αx·xᴴ, α REAL, one
//! triangle stored.
//!
//! Implementation: column-zaxpy over the stored strict segment
//! (column j's scalar is α·conj(`x[j]`)); the diagonal update
//! re(`x[j]`·t) is real and is stored with imaginary part exactly 0 —
//! reference `zher`'s DBLE() convention, kept as the layer's
//! Hermitian invariant (stored diagonals stay exactly real).

use super::super::check_mat;
use crate::{Error, Uplo};
use crate::c64::C64;
use crate::L1::zaxpy_k;

/// `A ← αxxᴴ + A` for a Hermitian `A` of order `n`, the triangle named
/// by `uplo` stored, at leading dimension `lda`; `x` has `n` elements.
///
/// Errors: `lda < max(1, n)` (argument 6); `x` or `a` shorter than its
/// extent (arguments 4, 5).
pub fn zher(uplo: Uplo, n: usize, alpha: f64, x: &[C64], a: &mut [C64], lda: usize) -> Result<(), Error> {
	const R: &str = "zher";
	crate::error::vec(R, 4, x.len(), n)?;
	crate::error::mat(R, 5, 6, a.len(), n, n, lda)?;
	if n == 0 {
		return Ok(());
	}
	if alpha == 0.0 {
		return Ok(()); // reference: nothing is read or written
	}
	zher_k(alpha, n, a, lda, uplo.is_upper(), &x[..n]);
	Ok(())
}

/// A ← A + αx·xᴴ with real α. A is Hermitian n×n at column stride
/// `cs`, `upper` (or lower) triangle stored.
pub(crate) fn zher_k(alpha: f64, n: usize, a: &mut [C64], cs: usize, upper: bool, x: &[C64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "zher: x length mismatch");
	for j in 0..n {
		let cj = j * cs;
		let t = x[j].conj().scale(alpha);
		if upper {
			zaxpy_k(t, &x[..j], &mut a[cj..cj + j]);
			a[cj + j] = C64::new(a[cj + j].re + (x[j] * t).re, 0.0);
		} else {
			a[cj + j] = C64::new(a[cj + j].re + (x[j] * t).re, 0.0);
			zaxpy_k(t, &x[j + 1..], &mut a[cj + j + 1..cj + n]);
		}
	}
}
