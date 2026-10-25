//! `dsyr2k` — symmetric rank-2k update: C ← α(ABᵀ + BAᵀ) + βC, one
//! triangle of C stored.
//!
//! Implementation: truncated column-daxpy, 4-column fan-out (same shape as dsyrk) — four stored columns grouped so
//! each inner A and B column streams once per group over the common
//! triangle segment; ragged edges scalar in the same per-element
//! order (bit-for-bit tested). Within each inner index the A-sourced
//! add still precedes the B-sourced add per element, as in the plain
//! two-stream sweep.

use super::super::check_mat;
use crate::{Error, Trans, Uplo};
use crate::kernels::daxpy4;
use crate::L1::daxpy_k;
use crate::L2::dscale_y;

/// `C ← α·op(A)·op(B)ᵀ + α·op(B)·op(A)ᵀ + βC` with `op(A)` and
/// `op(B)` `n×k` for [`Trans::NoTrans`] and `k×n` otherwise; `C` is
/// symmetric of order `n` with the triangle named by `uplo` stored and
/// updated. `work` is scratch of at least [`dsyr2k_work`]`(trans, n, k)` elements.
///
/// The transposed form transposes `A` and `B` into the workspace first and
/// then runs the no-transpose kernel, so it equals the unflagged call on
/// an explicit copy bit for bit; `ConjTrans` is `Trans` for a real matrix.
///
/// Errors: a leading dimension below its row count (arguments 7, 9,
/// 12); `a`, `b` or `c` shorter than its extent (arguments 6, 8, 11);
/// `work` shorter than the query (argument 13).
#[allow(clippy::too_many_arguments)]
pub fn dsyr2k(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f64, a: &[f64], lda: usize, b: &[f64], ldb: usize, beta: f64, c: &mut [f64], ldc: usize, work: &mut [f64]) -> Result<(), Error> {
	const R: &str = "dsyr2k";
	let (ar, ac) = if trans == Trans::NoTrans { (n, k) } else { (k, n) };
	crate::error::mat(R, 6, 7, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 8, 9, b.len(), ar, ac, ldb)?;
	crate::error::mat(R, 11, 12, c.len(), n, n, ldc)?;
	crate::error::work(R, 13, work.len(), dsyr2k_work(trans, n, k))?;
	if n == 0 {
		return Ok(());
	}
	if alpha == 0.0 || k == 0 {
		// reference: the stored triangle of C ← βC without reading A or B
		for j in 0..n {
			let (lo, hi) = if uplo.is_upper() { (0, j + 1) } else { (j, n) };
			crate::L2::dscale_y(beta, &mut c[j * ldc + lo..j * ldc + hi]);
		}
		return Ok(());
	}
	// op(A) and op(B) are materialized in the workspace ahead of the
	// no-transpose kernel, so the transposed form is bit-identical to
	// the unflagged call on an explicit copy (tested)
	let (work, a, lda) = if trans == Trans::NoTrans {
		(work, a, lda)
	} else {
		let (ta, rest) = work.split_at_mut(n * k);
		super::super::transpose_into(a, lda, k, n, ta, |x| x);
		(rest, &*ta, n)
	};
	let (_work, b, ldb) = if trans == Trans::NoTrans {
		(work, b, ldb)
	} else {
		let (tb, rest) = work.split_at_mut(n * k);
		super::super::transpose_into(b, ldb, k, n, tb, |x| x);
		(rest, &*tb, n)
	};
	dsyr2k_k(alpha, n, k, a, lda, b, ldb, beta, c, ldc, uplo.is_upper());
	Ok(())
}

/// Elements of scratch [`dsyr2k`] needs for these flags and dimensions:
/// `2·n·k` for the transposed form (the copies of `op(A)` and `op(B)`),
/// zero otherwise.
pub fn dsyr2k_work(trans: Trans, n: usize, k: usize) -> usize {
	if trans == Trans::NoTrans { 0 } else { 2 * n * k }
}

/// C ← α(ABᵀ + BAᵀ) + βC. A and B are n×k; C is n×n with the `upper`
/// (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsyr2k_k(
	alpha: f64,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	upper: bool,
) {
	check_mat(a.len(), n, k, acs);
	check_mat(b.len(), n, k, bcs);
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
		let (clo, chi) = if upper { (0, j + 1) } else { (j + 3, n) };
		let ragged =
			|u: usize| if upper { (chi, j + u + 1) } else { (j + u, clo) };
		for l in 0..k {
			let tb = [
				alpha * b[l * bcs + j],
				alpha * b[l * bcs + j + 1],
				alpha * b[l * bcs + j + 2],
				alpha * b[l * bcs + j + 3],
			];
			unsafe {
				daxpy4(
					a.as_ptr().add(l * acs + clo),
					tb,
					cp.add(j * ccs + clo),
					cp.add((j + 1) * ccs + clo),
					cp.add((j + 2) * ccs + clo),
					cp.add((j + 3) * ccs + clo),
					chi - clo,
				);
			}
			for (u, &tbu) in tb.iter().enumerate() {
				let (lo, hi) = ragged(u);
				for i in lo..hi {
					unsafe {
						*cp.add((j + u) * ccs + i) += a[l * acs + i] * tbu;
					}
				}
			}
			let ta = [
				alpha * a[l * acs + j],
				alpha * a[l * acs + j + 1],
				alpha * a[l * acs + j + 2],
				alpha * a[l * acs + j + 3],
			];
			unsafe {
				daxpy4(
					b.as_ptr().add(l * bcs + clo),
					ta,
					cp.add(j * ccs + clo),
					cp.add((j + 1) * ccs + clo),
					cp.add((j + 2) * ccs + clo),
					cp.add((j + 3) * ccs + clo),
					chi - clo,
				);
			}
			for (u, &tau) in ta.iter().enumerate() {
				let (lo, hi) = ragged(u);
				for i in lo..hi {
					unsafe {
						*cp.add((j + u) * ccs + i) += b[l * bcs + i] * tau;
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
			daxpy_k(alpha * b[l * bcs + j], &a[l * acs + lo..l * acs + hi], seg);
			daxpy_k(alpha * a[l * acs + j], &b[l * bcs + lo..l * bcs + hi], seg);
		}
		j += 1;
	}
}
