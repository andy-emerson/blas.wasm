//! `ctrmm` — triangular matrix multiply, in place: B ← αAB (left) or
//! B ← αBA (right), A triangular (complex).
//!
//! Implementation: the `dtrmm` shapes verbatim with complex scalars —
//! left side walks four B columns in lockstep through the `ctrmv`
//! column sweep sharing each A column read (`kernels::caxpy4`); right
//! side runs the in-group contributions in the original elimination
//! order, then fans out-of-group source columns via one stream each.
//! Same reorder disclosure as the real layers: LOWER-right keeps the plain
//! sweep's per-element order bit-for-bit; UPPER-right's in-group adds
//! precede the out-of-group adds — deterministic, documented, and
//! mirrored by the tests' scalar replay. Transpose/conjugate forms:
//! not built — no consumer yet.

use super::check_mat;
use crate::{Diag, Error, Side, Trans, Uplo};
use crate::c32::C32;
use crate::kernels::caxpy4;
use crate::L1::{caxpy_k, cscal};
use crate::L2::ctrmv_k;

/// `B ← α·op(A)·B` ([`Side::Left`]) or `B ← α·B·op(A)` ([`Side::Right`])
/// for a triangular `A`, the triangle named by `uplo` stored, with a
/// unit diagonal when `diag` says so; `A` is `m×m` on the left and `n×n`
/// on the right, `B` is `m×n`, each at its own leading dimension.
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ` per `transa`. `work` is scratch of at
/// least [`ctrmm_work`]`(side, transa, m, n)` elements.
///
/// The transposed forms are not built yet (`STATUS.md`, todo S1):
/// `transa` other than [`Trans::NoTrans`] returns
/// [`ErrorKind::Unsupported`](crate::ErrorKind::Unsupported) on
/// argument 3.
///
/// Errors: a leading dimension below its row count (arguments 9, 11);
/// `a` or `b` shorter than its extent (arguments 8, 10); `work` shorter
/// than the query (argument 12).
#[allow(clippy::too_many_arguments)]
pub fn ctrmm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: C32, a: &[C32], lda: usize, b: &mut [C32], ldb: usize, work: &mut [C32]) -> Result<(), Error> {
	const R: &str = "ctrmm";
	if transa != Trans::NoTrans {
		return Err(crate::error::unsupported(R, 3));
	}
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 8, 9, a.len(), na, na, lda)?;
	crate::error::mat(R, 10, 11, b.len(), m, n, ldb)?;
	crate::error::work(R, 12, work.len(), ctrmm_work(side, transa, m, n))?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	match side {
		Side::Left => ctrmm_left_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb),
		Side::Right => ctrmm_right_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb),
	}
	Ok(())
}

/// Elements of scratch [`ctrmm`] needs for these dimensions. Zero: no form
/// built so far uses scratch.
pub fn ctrmm_work(_side: Side, _transa: Trans, _m: usize, _n: usize) -> usize {
	0
}

/// B ← αAB. A is m×m triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ctrmm_left_k(
	alpha: C32,
	m: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [C32],
	bcs: usize,
) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let mut j = 0usize;
	while j + 4 <= n {
		let bp = b.as_mut_ptr();
		let cols: [*mut C32; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// the ctrmv column sweep, four x vectors in lockstep sharing
		// each A column read
		if upper {
			for l in 0..m {
				let t = unsafe {
					[*cols[0].add(l), *cols[1].add(l), *cols[2].add(l), *cols[3].add(l)]
				};
				unsafe {
					caxpy4(a.as_ptr().add(l * acs), t, cols[0], cols[1], cols[2], cols[3], l);
				}
				if !unit {
					let d = a[l * acs + l];
					for (u, cu) in cols.iter().enumerate() {
						unsafe { *cu.add(l) = t[u] * d };
					}
				}
			}
		} else {
			for l in (0..m).rev() {
				let t = unsafe {
					[*cols[0].add(l), *cols[1].add(l), *cols[2].add(l), *cols[3].add(l)]
				};
				unsafe {
					caxpy4(
						a.as_ptr().add(l * acs + l + 1),
						t,
						cols[0].add(l + 1),
						cols[1].add(l + 1),
						cols[2].add(l + 1),
						cols[3].add(l + 1),
						m - l - 1,
					);
				}
				if !unit {
					let d = a[l * acs + l];
					for (u, cu) in cols.iter().enumerate() {
						unsafe { *cu.add(l) = t[u] * d };
					}
				}
			}
		}
		if alpha != C32::ONE {
			for u in 0..4 {
				cscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		j += 4;
	}
	while j < n {
		let col = &mut b[j * bcs..j * bcs + m];
		ctrmv_k(m, a, acs, upper, unit, col);
		if alpha != C32::ONE {
			cscal(alpha, col);
		}
		j += 1;
	}
}

/// B ← αBA. A is n×n triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ctrmm_right_k(
	alpha: C32,
	m: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [C32],
	bcs: usize,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	// result column j = α·Σ_{k in triangle} a`[k,j]`·B_orig[:,k]; column
	// order (descending for upper, ascending for lower) touches each
	// original column before anything overwrites it. Columns j and k
	// never alias (k ≠ j always), so the raw-pointer views are
	// disjoint.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const C32, m) };
	let do_col = |j: usize| {
		let diag = if unit { C32::ONE } else { a[j * acs + j] };
		cscal(alpha * diag, col_mut(j));
		let (lo, hi) = if upper { (0, j) } else { (j + 1, n) };
		for k in lo..hi {
			caxpy_k(alpha * a[j * acs + k], col_ref(k), col_mut(j));
		}
	};
	let fan_out = |gs: usize, k: usize| {
		let t = [
			alpha * a[gs * acs + k],
			alpha * a[(gs + 1) * acs + k],
			alpha * a[(gs + 2) * acs + k],
			alpha * a[(gs + 3) * acs + k],
		];
		unsafe {
			caxpy4(
				bp.add(k * bcs) as *const C32,
				t,
				bp.add(gs * bcs),
				bp.add((gs + 1) * bcs),
				bp.add((gs + 2) * bcs),
				bp.add((gs + 3) * bcs),
				m,
			);
		}
	};
	if upper {
		let r = n % 4;
		let mut gs = n;
		while gs >= r + 4 {
			gs -= 4;
			// in-group first, original descending order (sources still
			// original when read)
			for tc in (gs..gs + 4).rev() {
				let diag = if unit { C32::ONE } else { a[tc * acs + tc] };
				cscal(alpha * diag, col_mut(tc));
				for k in gs..tc {
					caxpy_k(alpha * a[tc * acs + k], col_ref(k), col_mut(tc));
				}
			}
			// out-of-group sources, one shared stream each
			for k in 0..gs {
				fan_out(gs, k);
			}
		}
		for j in (0..r).rev() {
			do_col(j);
		}
	} else {
		let mut gs = 0usize;
		while gs + 4 <= n {
			// in-group ascending: per-element add order stays fully
			// ascending — bit-identical to the plain sweep for lower
			for tc in gs..gs + 4 {
				let diag = if unit { C32::ONE } else { a[tc * acs + tc] };
				cscal(alpha * diag, col_mut(tc));
				for k in tc + 1..gs + 4 {
					caxpy_k(alpha * a[tc * acs + k], col_ref(k), col_mut(tc));
				}
			}
			for k in gs + 4..n {
				fan_out(gs, k);
			}
			gs += 4;
		}
		for j in gs..n {
			do_col(j);
		}
	}
}
