//! `dsymm` — symmetric matrix multiply: C ← αAB + βC (left) or
//! C ← αBA + βC (right), A symmetric with one triangle stored.
//!
//! Implementation: left side is `dsymv` per column of B/C; right side
//! is a 4-column fan-out column-daxpy sweep (each B
//! column streams once per four C columns — the dgemm `col4` shape —
//! with the symmetric element looked up across the stored triangle;
//! per-element rounding sequence identical to the plain sweep,
//! bit-for-bit tested).

use super::super::{check_mat, dsym_at};
use crate::{Error, Side, Uplo};
use crate::kernels::daxpy4;
use crate::L1::daxpy_k;
use crate::L2::{dscale_y, dsymv_k};

/// `C ← αAB + βC` ([`Side::Left`]) or `C ← αBA + βC` ([`Side::Right`])
/// for a symmetric `A` with the triangle named by `uplo` stored; `A` is
/// `m×m` on the left and `n×n` on the right, `B` and `C` are `m×n`, each
/// at its own leading dimension.
///
/// Errors: a leading dimension below its row count (arguments 7, 9,
/// 12); `a`, `b` or `c` shorter than its extent (arguments 6, 8, 11).
#[allow(clippy::too_many_arguments)]
pub fn dsymm(side: Side, uplo: Uplo, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize, b: &[f64], ldb: usize, beta: f64, c: &mut [f64], ldc: usize) -> Result<(), Error> {
	const R: &str = "dsymm";
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 6, 7, a.len(), na, na, lda)?;
	crate::error::mat(R, 8, 9, b.len(), m, n, ldb)?;
	crate::error::mat(R, 11, 12, c.len(), m, n, ldc)?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if alpha == 0.0 {
		// reference: C ← βC without reading A or B
		for j in 0..n {
			crate::L2::dscale_y(beta, &mut c[j * ldc..j * ldc + m]);
		}
		return Ok(());
	}
	match side {
		Side::Left => dsymm_left_k(alpha, m, n, a, lda, uplo.is_upper(), b, ldb, beta, c, ldc),
		Side::Right => dsymm_right_k(alpha, m, n, a, lda, uplo.is_upper(), b, ldb, beta, c, ldc),
	}
	Ok(())
}

/// C ← αAB + βC. A is m×m symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsymm_left_k(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	check_mat(b.len(), m, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		dsymv_k(alpha, m, a, acs, upper, &b[j * bcs..j * bcs + m], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// C ← αBA + βC. A is n×n symmetric (`upper` triangle stored), B and
/// C are m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsymm_right_k(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
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
			dscale_y(beta, col);
		}
		for k in 0..n {
			let t = [
				alpha * dsym_at(a, acs, upper, k, j),
				alpha * dsym_at(a, acs, upper, k, j + 1),
				alpha * dsym_at(a, acs, upper, k, j + 2),
				alpha * dsym_at(a, acs, upper, k, j + 3),
			];
			unsafe {
				daxpy4(
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
		dscale_y(beta, cj);
		for k in 0..n {
			daxpy_k(alpha * dsym_at(a, acs, upper, k, j), &b[k * bcs..k * bcs + m], cj);
		}
		j += 1;
	}
}
