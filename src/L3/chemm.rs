//! `chemm` — Hermitian matrix multiply: C ← αAB + βC (left) or
//! C ← αBA + βC (right), A Hermitian with one triangle stored
//! (diagonal treated as real, per the storage convention).
//!
//! Implementation: the `dsymm` shapes — left side is `chemv` per
//! column of B/C (riding the fused column pass); right side is a
//! 4-column fan-out column-caxpy sweep with the Hermitian element
//! looked up across the stored triangle (`cher_at` — conjugating on
//! the reflected side, real on the diagonal); per-element rounding
//! sequence identical to the plain sweep, bit-for-bit tested.

use super::{check_mat, cher_at};
use crate::{Error, Side, Uplo};
use crate::c32::C32;
use crate::kernels::caxpy4;
use crate::L1::caxpy_k;
use crate::L2::{chemv_k, cscale_y};

/// `C ← αAB + βC` ([`Side::Left`]) or `C ← αBA + βC` ([`Side::Right`])
/// for a Hermitian `A` with the triangle named by `uplo` stored; `A` is
/// `m×m` on the left and `n×n` on the right, `B` and `C` are `m×n`, each
/// at its own leading dimension.
///
/// Errors: a leading dimension below its row count (arguments 7, 9,
/// 12); `a`, `b` or `c` shorter than its extent (arguments 6, 8, 11).
#[allow(clippy::too_many_arguments)]
pub fn chemm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: C32, a: &[C32], lda: usize, b: &[C32], ldb: usize, beta: C32, c: &mut [C32], ldc: usize) -> Result<(), Error> {
	const R: &str = "chemm";
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
		Side::Left => chemm_left_k(alpha, m, n, a, lda, uplo.is_upper(), b, ldb, beta, c, ldc),
		Side::Right => chemm_right_k(alpha, m, n, a, lda, uplo.is_upper(), b, ldb, beta, c, ldc),
	}
	Ok(())
}

/// C ← αAB + βC. A is m×m Hermitian (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chemm_left_k(
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
		chemv_k(alpha, m, a, acs, upper, &b[j * bcs..j * bcs + m], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// C ← αBA + βC. A is n×n Hermitian (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chemm_right_k(
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
				alpha * cher_at(a, acs, upper, k, j),
				alpha * cher_at(a, acs, upper, k, j + 1),
				alpha * cher_at(a, acs, upper, k, j + 2),
				alpha * cher_at(a, acs, upper, k, j + 3),
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
			caxpy_k(alpha * cher_at(a, acs, upper, k, j), &b[k * bcs..k * bcs + m], cj);
		}
		j += 1;
	}
}
