//! `dsyrk` — symmetric rank-k (Gram-matrix) update:
//! C ← αAAᵀ + βC, one triangle of C stored.
//!
//! Implementation: truncated column-daxpy, 4-column fan-out — four stored columns of C are grouped so each inner
//! A column streams once per group over the common triangle segment
//! (source traffic 4× down); the ≤3-row ragged edge per column is
//! handled scalar in the same per-element order, so results stay
//! bit-for-bit identical to the plain sweep (tested). The plain
//! (non-FMA) variant is the measured choice: fusing HARMED dsyrk on
//! all three campaign record step-1 draws.

use super::check_mat;
use crate::{Error, Trans, Uplo};
use crate::kernels::daxpy4;
use crate::L1::daxpy_k;
use crate::L2::dscale_y;

/// `C ← α·op(A)·op(A)ᵀ + βC` with `op(A)` `n×k` for [`Trans::NoTrans`]
/// and `k×n` otherwise; `C` is symmetric of order `n` with the triangle
/// named by `uplo` stored and updated. `work` is scratch of at least
/// [`dsyrk_work`]`(trans, n, k)` elements.
///
/// The transposed form transposes `A` into the workspace first and
/// then runs the no-transpose kernel, so it equals the unflagged call on
/// an explicit copy bit for bit; `ConjTrans` is `Trans` for a real matrix.
///
/// Errors: a leading dimension below its row count (arguments 7, 10);
/// `a` or `c` shorter than its extent (arguments 6, 9); `work` shorter
/// than the query (argument 11).
#[allow(clippy::too_many_arguments)]
pub fn dsyrk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f64, a: &[f64], lda: usize, beta: f64, c: &mut [f64], ldc: usize, work: &mut [f64]) -> Result<(), Error> {
	const R: &str = "dsyrk";
	let (ar, ac) = if trans == Trans::NoTrans { (n, k) } else { (k, n) };
	crate::error::mat(R, 6, 7, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 9, 10, c.len(), n, n, ldc)?;
	crate::error::work(R, 11, work.len(), dsyrk_work(trans, n, k))?;
	if n == 0 {
		return Ok(());
	}
	if alpha == 0.0 || k == 0 {
		// reference: the stored triangle of C ← βC without reading A
		for j in 0..n {
			let (lo, hi) = if uplo.is_upper() { (0, j + 1) } else { (j, n) };
			crate::L2::dscale_y(beta, &mut c[j * ldc + lo..j * ldc + hi]);
		}
		return Ok(());
	}
	// op(A) is materialized in the workspace ahead of the
	// no-transpose kernel, so the transposed form is bit-identical to
	// the unflagged call on an explicit copy (tested)
	let (work, a, lda) = if trans == Trans::NoTrans {
		(work, a, lda)
	} else {
		let (ta, rest) = work.split_at_mut(n * k);
		super::transpose_into(a, lda, k, n, ta, |x| x);
		(rest, &*ta, n)
	};
	let _ = work;
	dsyrk_k(alpha, n, k, a, lda, beta, c, ldc, uplo.is_upper());
	Ok(())
}

/// Elements of scratch [`dsyrk`] needs for these flags and dimensions:
/// `n·k` for the transposed form (the copy of `op(A)`), zero otherwise.
pub fn dsyrk_work(trans: Trans, n: usize, k: usize) -> usize {
	if trans == Trans::NoTrans { 0 } else { n * k }
}

/// C ← αAAᵀ + βC. A is n×k; C is n×n with the `upper` (or lower)
/// triangle stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsyrk_k(
	alpha: f64,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	beta: f64,
	c: &mut [f64],
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
			dscale_y(beta, seg);
		}
		// intersection of the four stored segments; the rest is the
		// ≤3-row ragged edge per column
		let (clo, chi) = if upper { (0, j + 1) } else { (j + 3, n) };
		for l in 0..k {
			let t = [
				alpha * a[l * acs + j],
				alpha * a[l * acs + j + 1],
				alpha * a[l * acs + j + 2],
				alpha * a[l * acs + j + 3],
			];
			unsafe {
				daxpy4(
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
						*cp.add((j + u) * ccs + i) += a[l * acs + i] * tu;
					}
				}
			}
		}
		j += 4;
	}
	while j < n {
		let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
		let seg = &mut c[j * ccs + lo..j * ccs + hi];
		dscale_y(beta, seg);
		for l in 0..k {
			daxpy_k(alpha * a[l * acs + j], &a[l * acs + lo..l * acs + hi], seg);
		}
		j += 1;
	}
}
