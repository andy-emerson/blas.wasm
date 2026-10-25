//! `zsymm` — complex SYMMETRIC matrix multiply: C ← αAB + βC (left) or
//! C ← αBA + βC (right), A symmetric (not Hermitian) with one triangle
//! stored — reference BLAS's `zsymm`.
//!
//! Implementation: the `zhemm` shapes without the conjugation — left
//! side is a symmetric matrix-vector pass per column of B/C (the fused
//! column pass of `zhemv` on the unconjugated product form, a
//! crate-private kernel here since reference BLAS has no symmetric
//! complex matrix-vector routine); right side is a 4-column fan-out
//! column-axpy sweep with the symmetric element looked up across the
//! stored triangle (`zsym_at`, the mirror unconjugated, the diagonal an
//! ordinary complex element) whose per-element rounding sequence is
//! identical to the plain sweep, bit-for-bit tested; the left side's
//! fused pass re-folds its reduction and is bounds-tested.

use super::{check_mat, zsym_at};
use crate::{Error, Side, Uplo};
use crate::c64::C64;
use crate::kernels::{zaxpy4, zaxpy_dotu, zaxpy_dotu4};
use crate::L1::zaxpy_k;
use crate::L2::zscale_y;

/// `C ← αAB + βC` ([`Side::Left`]) or `C ← αBA + βC` ([`Side::Right`])
/// for a symmetric (not Hermitian) `A` with the triangle named by `uplo` stored; `A` is
/// `m×m` on the left and `n×n` on the right, `B` and `C` are `m×n`, each
/// at its own leading dimension.
///
/// Errors: a leading dimension below its row count (arguments 7, 9,
/// 12); `a`, `b` or `c` shorter than its extent (arguments 6, 8, 11).
#[allow(clippy::too_many_arguments)]
pub fn zsymm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: C64, a: &[C64], lda: usize, b: &[C64], ldb: usize, beta: C64, c: &mut [C64], ldc: usize) -> Result<(), Error> {
	const R: &str = "zsymm";
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 6, 7, a.len(), na, na, lda)?;
	crate::error::mat(R, 8, 9, b.len(), m, n, ldb)?;
	crate::error::mat(R, 11, 12, c.len(), m, n, ldc)?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if alpha == C64::ZERO {
		// reference: C ← βC without reading A or B
		for j in 0..n {
			crate::L2::zscale_y(beta, &mut c[j * ldc..j * ldc + m]);
		}
		return Ok(());
	}
	match side {
		Side::Left => zsymm_left_k(alpha, m, n, a, lda, uplo.is_upper(), b, ldb, beta, c, ldc),
		Side::Right => zsymm_right_k(alpha, m, n, a, lda, uplo.is_upper(), b, ldb, beta, c, ldc),
	}
	Ok(())
}

/// C ← αAB + βC. A is m×m symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zsymm_left_k(
	alpha: C64,
	m: usize,
	n: usize,
	a: &[C64],
	acs: usize,
	upper: bool,
	b: &[C64],
	bcs: usize,
	beta: C64,
	c: &mut [C64],
	ccs: usize,
) {
	check_mat(b.len(), m, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		zsymv_k(alpha, m, a, acs, upper, &b[j * bcs..j * bcs + m], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// C ← αBA + βC. A is n×n symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zsymm_right_k(
	alpha: C64,
	m: usize,
	n: usize,
	a: &[C64],
	acs: usize,
	upper: bool,
	b: &[C64],
	bcs: usize,
	beta: C64,
	c: &mut [C64],
	ccs: usize,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		let cp = c.as_mut_ptr();
		for u in 0..4 {
			let col = unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs), m) };
			zscale_y(beta, col);
		}
		for k in 0..n {
			let t = [
				alpha * zsym_at(a, acs, upper, k, j),
				alpha * zsym_at(a, acs, upper, k, j + 1),
				alpha * zsym_at(a, acs, upper, k, j + 2),
				alpha * zsym_at(a, acs, upper, k, j + 3),
			];
			unsafe {
				zaxpy4(
					b.as_ptr().add(k * bcs),
					t,
					cp.add(j * ccs),
					cp.add((j + 1) * ccs),
					cp.add((j + 2) * ccs),
					cp.add((j + 3) * ccs),
					m,
				);
			}
		}
		j += 4;
	}
	while j < n {
		let cj = &mut c[j * ccs..j * ccs + m];
		zscale_y(beta, cj);
		for k in 0..n {
			zaxpy_k(alpha * zsym_at(a, acs, upper, k, j), &b[k * bcs..k * bcs + m], cj);
		}
		j += 1;
	}
}

/// y ← αAx + βy for a complex SYMMETRIC A (`upper` triangle stored),
/// the `zhemv` fused column pass with the unconjugated product form and
/// the diagonal taken as an ordinary complex element. Crate-private:
/// reference BLAS has no symmetric complex matrix-vector routine.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zsymv_k(
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
	assert_eq!(x.len(), n, "zsymv: x length mismatch");
	assert_eq!(y.len(), n, "zsymv: y length mismatch");
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
			let mut d = unsafe { zaxpy_dotu4(cols, t, x.as_ptr(), y.as_mut_ptr(), j) };
			for u in 0..4 {
				let cj = (j + u) * cs;
				for i in j..j + u {
					y[i] = y[i] + t[u] * a[cj + i];
					d[u] = d[u] + a[cj + i] * x[i];
				}
				y[j + u] = y[j + u] + t[u] * a[cj + j + u] + alpha * d[u];
			}
		} else {
			// common strict segment rows j+4..n; ragged rows j+u+1..j+4
			let mut d = unsafe {
				zaxpy_dotu4(
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
					d[u] = d[u] + a[cj + i] * x[i];
				}
				y[j + u] = y[j + u] + t[u] * a[cj + j + u] + alpha * d[u];
			}
		}
		j += 4;
	}
	while j < n {
		let cj = j * cs;
		let t = alpha * x[j];
		let d = if upper {
			unsafe { zaxpy_dotu(a.as_ptr().add(cj), t, x.as_ptr(), y.as_mut_ptr(), j) }
		} else {
			unsafe {
				zaxpy_dotu(
					a.as_ptr().add(cj + j + 1),
					t,
					x.as_ptr().add(j + 1),
					y.as_mut_ptr().add(j + 1),
					n - j - 1,
				)
			}
		};
		y[j] = y[j] + t * a[cj + j] + alpha * d;
		j += 1;
	}
}
