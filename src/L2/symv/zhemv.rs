//! `zhemv` — Hermitian matrix × vector: y ← αAx + βy, A Hermitian
//! with only one triangle stored (`A[j,i]` = conj(`A[i,j]`); diagonal
//! imaginary parts are ignored, per the LAPACK storage convention).
//!
//! Implementation: 4-column grouped fused pass (close-out race)
//! — four stored columns share ONE stream over x and y
//! (`kernels::zaxpy_dotc4`: `y[i]` += Σᵤ tᵤ·`aᵤ[i]` while each
//! accᵤ += conj(`aᵤ[i]`)·`x[i]`), the dsymv grouping at complex lane
//! geometry; tail columns use the single-column fused pass
//! (`kernels::zaxpy_dotc`). Raced against the single-column shape on
//! both reference draws and WON (3.19–3.45 vs 3.58–3.92 ms at
//! n=2048, ~9–13%) — the same lever LOST for c32 (see `chemv`), so
//! each type ships its own winner. The diagonal contributes
//! t·re(`a[j,j]`) (real by convention). Accumulation order is the
//! grouped pass's own — zhemv is bounds-tested, not bit-locked;
//! cross-target determinism holds through the lane emulation as
//! everywhere else.

use super::super::{check_mat, zscale_y};
use crate::{Error, Uplo};
use crate::c64::C64;
use crate::kernels::{zaxpy_dotc, zaxpy_dotc4};

/// `y ← αAx + βy` for a Hermitian `A` of order `n` with the triangle
/// named by `uplo` stored, at leading dimension `lda`; `x` and `y` have
/// `n` elements. Longer slices are used up to that length.
///
/// Errors: `lda < max(1, n)` (argument 5); `a`, `x` or `y` shorter than
/// its extent (arguments 4, 6, 8).
#[allow(clippy::too_many_arguments)]
pub fn zhemv(uplo: Uplo, n: usize, alpha: C64, a: &[C64], lda: usize, x: &[C64], beta: C64, y: &mut [C64]) -> Result<(), Error> {
	const R: &str = "zhemv";
	crate::error::mat(R, 4, 5, a.len(), n, n, lda)?;
	crate::error::vec(R, 6, x.len(), n)?;
	crate::error::vec(R, 8, y.len(), n)?;
	if n == 0 {
		return Ok(());
	}
	if alpha == C64::ZERO {
		// reference: y ← βy without reading A or x
		crate::L2::zscale_y(beta, &mut y[..n]);
		return Ok(());
	}
	zhemv_k(alpha, n, a, lda, uplo.is_upper(), &x[..n], beta, &mut y[..n]);
	Ok(())
}

/// y ← αAx + βy, A Hermitian n×n at column stride `cs`, with the
/// `upper` (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zhemv_k(
	alpha: C64,
	n: usize,
	a: &[C64],
	cs: usize,
	upper: bool,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "zhemv: x length mismatch");
	assert_eq!(y.len(), n, "zhemv: y length mismatch");
	zscale_y(beta, y);
	let ap = a.as_ptr();
	let mut j = 0usize;
	while j + 4 <= n {
		let t = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
		let cols = unsafe {
			[ap.add(j * cs), ap.add((j + 1) * cs), ap.add((j + 2) * cs), ap.add((j + 3) * cs)]
		};
		if upper {
			// common strict segment rows 0..j fused for all four; then
			// the ragged rows j..j+u and the diagonal, scalar per column
			let mut d = unsafe { zaxpy_dotc4(cols, t, x.as_ptr(), y.as_mut_ptr(), j) };
			for u in 0..4 {
				let cj = (j + u) * cs;
				for i in j..j + u {
					y[i] = y[i] + t[u] * a[cj + i];
					d[u] = d[u] + a[cj + i].conj() * x[i];
				}
				y[j + u] = y[j + u] + t[u].scale(a[cj + j + u].re) + alpha * d[u];
			}
		} else {
			// common strict segment rows j+4..n; ragged rows j+u+1..j+4
			let mut d = unsafe {
				zaxpy_dotc4(
					[cols[0].add(j + 4), cols[1].add(j + 4), cols[2].add(j + 4), cols[3].add(j + 4)],
					t,
					x.as_ptr().add(j + 4),
					y.as_mut_ptr().add(j + 4),
					n - j - 4,
				)
			};
			for u in 0..4 {
				let cj = (j + u) * cs;
				for i in j + u + 1..j + 4 {
					y[i] = y[i] + t[u] * a[cj + i];
					d[u] = d[u] + a[cj + i].conj() * x[i];
				}
				y[j + u] = y[j + u] + t[u].scale(a[cj + j + u].re) + alpha * d[u];
			}
		}
		j += 4;
	}
	while j < n {
		let cj = j * cs;
		let t = alpha * x[j];
		let d = if upper {
			unsafe { zaxpy_dotc(a.as_ptr().add(cj), t, x.as_ptr(), y.as_mut_ptr(), j) }
		} else {
			unsafe {
				zaxpy_dotc(
					a.as_ptr().add(cj + j + 1),
					t,
					x.as_ptr().add(j + 1),
					y.as_mut_ptr().add(j + 1),
					n - j - 1,
				)
			}
		};
		y[j] = y[j] + t.scale(a[cj + j].re) + alpha * d;
		j += 1;
	}
}
