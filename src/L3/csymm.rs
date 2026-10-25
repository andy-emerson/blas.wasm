//! `csymm` — complex SYMMETRIC matrix multiply: C ← αAB + βC (left) or
//! C ← αBA + βC (right), A symmetric (not Hermitian) with one triangle
//! stored — reference BLAS's `csymm`.
//!
//! Implementation: the `chemm` shapes without the conjugation — left
//! side is a symmetric matrix-vector pass per column of B/C (the fused
//! column pass of `chemv` on the unconjugated product form, a
//! crate-private kernel here since reference BLAS has no symmetric
//! complex matrix-vector routine); right side is a 4-column fan-out
//! column-axpy sweep with the symmetric element looked up across the
//! stored triangle (`csym_at`, the mirror unconjugated, the diagonal an
//! ordinary complex element) whose per-element rounding sequence is
//! identical to the plain sweep, bit-for-bit tested; the left side's
//! fused pass re-folds its reduction and is bounds-tested.

use super::{check_mat, csym_at};
use crate::{Error, Side, Uplo};
use crate::c32::C32;
use crate::kernels::{caxpy4, caxpy_dotu};
use crate::L1::caxpy_k;
use crate::L2::cscale_y;

/// `C ← αAB + βC` ([`Side::Left`]) or `C ← αBA + βC` ([`Side::Right`])
/// for a symmetric (not Hermitian) `A` with the triangle named by `uplo` stored; `A` is
/// `m×m` on the left and `n×n` on the right, `B` and `C` are `m×n`, each
/// at its own leading dimension.
///
/// Errors: a leading dimension below its row count (arguments 7, 9,
/// 12); `a`, `b` or `c` shorter than its extent (arguments 6, 8, 11).
#[allow(clippy::too_many_arguments)]
pub fn csymm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: C32, a: &[C32], lda: usize, b: &[C32], ldb: usize, beta: C32, c: &mut [C32], ldc: usize) -> Result<(), Error> {
	const R: &str = "csymm";
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 6, 7, a.len(), na, na, lda)?;
	crate::error::mat(R, 8, 9, b.len(), m, n, ldb)?;
	crate::error::mat(R, 11, 12, c.len(), m, n, ldc)?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if alpha == C32::ZERO {
		// reference: C ← βC without reading A or B
		for j in 0..n {
			crate::L2::cscale_y(beta, &mut c[j * ldc..j * ldc + m]);
		}
		return Ok(());
	}
	match side {
		Side::Left => csymm_left_k(alpha, m, n, a, lda, uplo.is_upper(), b, ldb, beta, c, ldc),
		Side::Right => csymm_right_k(alpha, m, n, a, lda, uplo.is_upper(), b, ldb, beta, c, ldc),
	}
	Ok(())
}

/// C ← αAB + βC. A is m×m symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn csymm_left_k(
	alpha: C32,
	m: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	upper: bool,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
) {
	check_mat(b.len(), m, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		csymv_k(alpha, m, a, acs, upper, &b[j * bcs..j * bcs + m], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// C ← αBA + βC. A is n×n symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn csymm_right_k(
	alpha: C32,
	m: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	upper: bool,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
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
			cscale_y(beta, col);
		}
		for k in 0..n {
			let t = [
				alpha * csym_at(a, acs, upper, k, j),
				alpha * csym_at(a, acs, upper, k, j + 1),
				alpha * csym_at(a, acs, upper, k, j + 2),
				alpha * csym_at(a, acs, upper, k, j + 3),
			];
			unsafe {
				caxpy4(
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
		cscale_y(beta, cj);
		for k in 0..n {
			caxpy_k(alpha * csym_at(a, acs, upper, k, j), &b[k * bcs..k * bcs + m], cj);
		}
		j += 1;
	}
}

/// y ← αAx + βy for a complex SYMMETRIC A (`upper` triangle stored),
/// the `chemv` fused column pass with the unconjugated product form and
/// the diagonal taken as an ordinary complex element. Crate-private:
/// reference BLAS has no symmetric complex matrix-vector routine.
#[allow(clippy::too_many_arguments)]
pub(crate) fn csymv_k(
	alpha: C32,
	n: usize,
	a: &[C32],
	cs: usize,
	upper: bool,
	x: &[C32],
	beta: C32,
	y: &mut [C32],
) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "csymv: x length mismatch");
	assert_eq!(y.len(), n, "csymv: y length mismatch");
	cscale_y(beta, y);
	for j in 0..n {
		let cj = j * cs;
		let t = alpha * x[j];
		let d = if upper {
			unsafe { caxpy_dotu(a.as_ptr().add(cj), t, x.as_ptr(), y.as_mut_ptr(), j) }
		} else {
			unsafe {
				caxpy_dotu(
					a.as_ptr().add(cj + j + 1),
					t,
					x.as_ptr().add(j + 1),
					y.as_mut_ptr().add(j + 1),
					n - j - 1,
				)
			}
		};
		y[j] = y[j] + t * a[cj + j] + alpha * d;
	}
}
