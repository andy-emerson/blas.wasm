//! `ssyrk` — symmetric rank-k (Gram-matrix) update:
//! C ← αAAᵀ + βC, one triangle of C stored.
//!
//! Implementation: truncated column-saxpy, 4-column fan-out — four stored columns of C are grouped so each inner
//! A column streams once per group over the common triangle segment
//! (source traffic 4× down); the ≤3-row ragged edge per column is
//! handled scalar in the same per-element order, so results stay
//! bit-for-bit identical to the plain sweep (tested). The plain
//! (non-FMA) variant is the measured choice: fusing HARMED ssyrk on
//! all three campaign record step-1 draws.

use super::check_mat;
use crate::{Error, Trans, Uplo};
use crate::kernels::saxpy4;
use crate::L1::saxpy_k;
use crate::L2::sscale_y;

/// `C ← α·op(A)·op(A)ᵀ + βC` with `op(A)` `n×k` for [`Trans::NoTrans`]
/// and `k×n` otherwise; `C` is symmetric of order `n` with the triangle
/// named by `uplo` stored and updated. `work` is scratch of at least
/// [`ssyrk_work`]`(trans, n, k)` elements.
///
/// The transposed form transposes `A` into the workspace first and
/// then runs the no-transpose kernel, so it equals the unflagged call on
/// an explicit copy bit for bit; `ConjTrans` is `Trans` for a real matrix.
///
/// Errors: a leading dimension below its row count (arguments 7, 10);
/// `a` or `c` shorter than its extent (arguments 6, 9); `work` shorter
/// than the query (argument 11).
#[allow(clippy::too_many_arguments)]
pub fn ssyrk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize, beta: f32, c: &mut [f32], ldc: usize, work: &mut [f32]) -> Result<(), Error> {
	const R: &str = "ssyrk";
	let (ar, ac) = if trans == Trans::NoTrans { (n, k) } else { (k, n) };
	crate::error::mat(R, 6, 7, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 9, 10, c.len(), n, n, ldc)?;
	crate::error::work(R, 11, work.len(), ssyrk_work(trans, n, k))?;
	if n == 0 {
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
	ssyrk_k(alpha, n, k, a, lda, beta, c, ldc, uplo.is_upper());
	Ok(())
}

/// Elements of scratch [`ssyrk`] needs for these flags and dimensions:
/// `n·k` for the transposed form (the copy of `op(A)`), zero otherwise.
pub fn ssyrk_work(trans: Trans, n: usize, k: usize) -> usize {
	if trans == Trans::NoTrans { 0 } else { n * k }
}

/// C ← αAAᵀ + βC. A is n×k; C is n×n with the `upper` (or lower)
/// triangle stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ssyrk_k(
	alpha: f32,
	n: usize,
	k: usize,
	a: &[f32],
	acs: usize,
	beta: f32,
	c: &mut [f32],
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
			sscale_y(beta, seg);
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
				saxpy4(
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
		sscale_y(beta, seg);
		for l in 0..k {
			saxpy_k(alpha * a[l * acs + j], &a[l * acs + lo..l * acs + hi], seg);
		}
		j += 1;
	}
}
