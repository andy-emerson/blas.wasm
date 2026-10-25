//! `cherk` — Hermitian rank-k update: C ← αAAᴴ + βC, α and β REAL,
//! one triangle of C stored.
//!
//! Implementation: the `dsyrk` shape — truncated column-caxpy,
//! 4-column fan-out over the common triangle segment, ragged edges
//! scalar in the same per-element order (bit-for-bit tested). Column
//! j's inner-l scalar is α·conj(`A[j,l]`). After each column's
//! accumulation the diagonal's imaginary part is set to exactly 0 —
//! reference `cherk`'s DBLE() convention (the products' imaginary
//! roundoff is discarded), the layer's Hermitian invariant.

use super::check_mat;
use crate::{Error, Trans, Uplo};
use crate::c32::C32;
use crate::kernels::caxpy4;
use crate::L1::caxpy_k;
use crate::L2::csscale_y;

/// `C ← α·op(A)·op(A)ᴴ + βC` with `op(A)` `n×k` for [`Trans::NoTrans`]
/// and `k×n` otherwise; `C` is Hermitian of order `n` with the triangle
/// named by `uplo` stored and updated. `work` is scratch of at least
/// [`cherk_work`]`(n, k)` elements.
///
/// [`Trans::Trans`] is not a valid form for a Hermitian update and
/// returns [`ErrorKind::Argument`](crate::ErrorKind::Argument) on argument 2;
/// [`Trans::ConjTrans`] is not built yet (`STATUS.md`, todo S1) and returns
/// [`ErrorKind::Unsupported`](crate::ErrorKind::Unsupported).
///
/// Errors: a leading dimension below its row count (arguments 7, 10);
/// `a` or `c` shorter than its extent (arguments 6, 9); `work` shorter
/// than the query (argument 11).
#[allow(clippy::too_many_arguments)]
pub fn cherk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f32, a: &[C32], lda: usize, beta: f32, c: &mut [C32], ldc: usize, work: &mut [C32]) -> Result<(), Error> {
	const R: &str = "cherk";
	if trans == Trans::Trans {
		return Err(crate::error::argument(R, 2));
	}
	if trans != Trans::NoTrans {
		return Err(crate::error::unsupported(R, 2));
	}
	let (ar, ac) = if trans == Trans::NoTrans { (n, k) } else { (k, n) };
	crate::error::mat(R, 6, 7, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 9, 10, c.len(), n, n, ldc)?;
	crate::error::work(R, 11, work.len(), cherk_work(n, k))?;
	cherk_k(alpha, n, k, a, lda, beta, c, ldc, uplo.is_upper());
	Ok(())
}

/// Elements of scratch [`cherk`] needs for these dimensions. Zero: no form
/// built so far uses scratch.
pub fn cherk_work(_n: usize, _k: usize) -> usize {
	0
}

/// C ← αAAᴴ + βC with real α, β. A is n×k; C is n×n with the `upper`
/// (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cherk_k(
	alpha: f32,
	n: usize,
	k: usize,
	a: &[C32],
	acs: usize,
	beta: f32,
	c: &mut [C32],
	ccs: usize,
	upper: bool,
) {
	check_mat(a.len(), n, k, acs);
	check_mat(c.len(), n, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		let cp = c.as_mut_ptr();
		for u in 0..4 {
			let (lo, hi) = if upper { (0, j + u + 1) } else { (j + u, n) };
			let seg =
				unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs + lo), hi - lo) };
			csscale_y(beta, seg);
		}
		// intersection of the four stored segments; the rest is the
		// ≤3-row ragged edge per column
		let (clo, chi) = if upper { (0, j + 1) } else { (j + 3, n) };
		for l in 0..k {
			let t = [
				a[l * acs + j].conj().scale(alpha),
				a[l * acs + j + 1].conj().scale(alpha),
				a[l * acs + j + 2].conj().scale(alpha),
				a[l * acs + j + 3].conj().scale(alpha),
			];
			unsafe {
				caxpy4(
					a.as_ptr().add(l * acs + clo),
					t,
					cp.add(j * ccs + clo),
					cp.add((j + 1) * ccs + clo),
					cp.add((j + 2) * ccs + clo),
					cp.add((j + 3) * ccs + clo),
					chi - clo,
				);
			}
			for (u, &tu) in t.iter().enumerate() {
				let (lo, hi) = if upper { (chi, j + u + 1) } else { (j + u, clo) };
				for i in lo..hi {
					unsafe {
						let p = cp.add((j + u) * ccs + i);
						*p = *p + a[l * acs + i] * tu;
					}
				}
			}
		}
		// Hermitian invariant: the four diagonals end exactly real
		for u in 0..4 {
			let d = &mut c[(j + u) * ccs + j + u];
			*d = C32::new(d.re, 0.0);
		}
		j += 4;
	}
	while j < n {
		let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
		let seg = &mut c[j * ccs + lo..j * ccs + hi];
		csscale_y(beta, seg);
		for l in 0..k {
			caxpy_k(
				a[l * acs + j].conj().scale(alpha),
				&a[l * acs + lo..l * acs + hi],
				&mut c[j * ccs + lo..j * ccs + hi],
			);
		}
		let d = &mut c[j * ccs + j];
		*d = C32::new(d.re, 0.0);
		j += 1;
	}
}
