//! `zher2k` — Hermitian rank-2k update: C ← αABᴴ + conj(α)BAᴴ + βC,
//! β REAL, one triangle of C stored.
//!
//! Implementation: the `dsyr2k` shape — truncated column-zaxpy,
//! 4-column fan-out over the common triangle segment for both the
//! A-sourced and B-sourced streams, ragged edges scalar in the same
//! per-element order (bit-for-bit tested). Column j's inner-l scalars
//! are α·conj(`B[j,l]`) (applied to A's column) then conj(α·`A[j,l]`)
//! (applied to B's column); within each inner index the A-sourced add
//! precedes the B-sourced add per element, as in the plain two-stream
//! sweep. Diagonals end exactly real (reference DBLE() convention,
//! the layer's Hermitian invariant).

use super::check_mat;
use crate::{Error, Trans, Uplo};
use crate::c64::C64;
use crate::kernels::zaxpy4;
use crate::L1::zaxpy_k;
use crate::L2::zdscale_y;

/// `C ← α·op(A)·op(B)ᴴ + conj(α)·op(B)·op(A)ᴴ + βC` with `op(A)` and
/// `op(B)` `n×k` for [`Trans::NoTrans`] and `k×n` otherwise; `C` is
/// Hermitian of order `n` with the triangle named by `uplo` stored and
/// updated. `work` is scratch of at least [`zher2k_work`]`(n, k)` elements.
///
/// [`Trans::Trans`] is not a valid form for a Hermitian update and
/// returns [`ErrorKind::Argument`](crate::ErrorKind::Argument) on argument 2;
/// [`Trans::ConjTrans`] is not built yet (`STATUS.md`, todo S1) and returns
/// [`ErrorKind::Unsupported`](crate::ErrorKind::Unsupported).
///
/// Errors: a leading dimension below its row count (arguments 7, 9,
/// 12); `a`, `b` or `c` shorter than its extent (arguments 6, 8, 11);
/// `work` shorter than the query (argument 13).
#[allow(clippy::too_many_arguments)]
pub fn zher2k(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: C64, a: &[C64], lda: usize, b: &[C64], ldb: usize, beta: f64, c: &mut [C64], ldc: usize, work: &mut [C64]) -> Result<(), Error> {
	const R: &str = "zher2k";
	if trans == Trans::Trans {
		return Err(crate::error::argument(R, 2));
	}
	if trans != Trans::NoTrans {
		return Err(crate::error::unsupported(R, 2));
	}
	let (ar, ac) = if trans == Trans::NoTrans { (n, k) } else { (k, n) };
	crate::error::mat(R, 6, 7, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 8, 9, b.len(), ar, ac, ldb)?;
	crate::error::mat(R, 11, 12, c.len(), n, n, ldc)?;
	crate::error::work(R, 13, work.len(), zher2k_work(n, k))?;
	if n == 0 {
		return Ok(());
	}
	zher2k_k(alpha, n, k, a, lda, b, ldb, beta, c, ldc, uplo.is_upper());
	Ok(())
}

/// Elements of scratch [`zher2k`] needs for these dimensions. Zero: no form
/// built so far uses scratch.
pub fn zher2k_work(_n: usize, _k: usize) -> usize {
	0
}

/// C ← αABᴴ + conj(α)BAᴴ + βC with real β. A and B are n×k; C is n×n
/// with the `upper` (or lower) triangle stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zher2k_k(
	alpha: C64,
	n: usize,
	k: usize,
	a: &[C64],
	acs: usize,
	b: &[C64],
	bcs: usize,
	beta: f64,
	c: &mut [C64],
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
			zdscale_y(beta, seg);
		}
		let (clo, chi) = if upper { (0, j + 1) } else { (j + 3, n) };
		let ragged = |u: usize| if upper { (chi, j + u + 1) } else { (j + u, clo) };
		for l in 0..k {
			let tb = [
				alpha * b[l * bcs + j].conj(),
				alpha * b[l * bcs + j + 1].conj(),
				alpha * b[l * bcs + j + 2].conj(),
				alpha * b[l * bcs + j + 3].conj(),
			];
			unsafe {
				zaxpy4(
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
						let p = cp.add((j + u) * ccs + i);
						*p = *p + a[l * acs + i] * tbu;
					}
				}
			}
			let ta = [
				(alpha * a[l * acs + j]).conj(),
				(alpha * a[l * acs + j + 1]).conj(),
				(alpha * a[l * acs + j + 2]).conj(),
				(alpha * a[l * acs + j + 3]).conj(),
			];
			unsafe {
				zaxpy4(
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
						let p = cp.add((j + u) * ccs + i);
						*p = *p + b[l * bcs + i] * tau;
					}
				}
			}
		}
		// Hermitian invariant: the four diagonals end exactly real
		for u in 0..4 {
			let d = &mut c[(j + u) * ccs + j + u];
			*d = C64::new(d.re, 0.0);
		}
		j += 4;
	}
	while j < n {
		let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
		let seg = &mut c[j * ccs + lo..j * ccs + hi];
		zdscale_y(beta, seg);
		for l in 0..k {
			zaxpy_k(
				alpha * b[l * bcs + j].conj(),
				&a[l * acs + lo..l * acs + hi],
				&mut c[j * ccs + lo..j * ccs + hi],
			);
			zaxpy_k(
				(alpha * a[l * acs + j]).conj(),
				&b[l * bcs + lo..l * bcs + hi],
				&mut c[j * ccs + lo..j * ccs + hi],
			);
		}
		let d = &mut c[j * ccs + j];
		*d = C64::new(d.re, 0.0);
		j += 1;
	}
}
