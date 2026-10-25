//! `dtrsm` — triangular solve, many right-hand sides, in place:
//! B ← αA⁻¹B (left) or B ← αBA⁻¹ (right).
//!
//! Implementation: left side is divide-then-
//! column-daxpy — αscal, then four B columns walk the `dtrsv`
//! elimination in lockstep so each A column streams once per group of
//! four (A traffic 4× down; the columns are independent, so results
//! are bit-for-bit the plain dtrsv-per-column loop, tested). Right
//! side groups four destination columns: already-solved columns
//! outside the group fan out via one shared stream each, then the
//! in-group elimination runs in the original order (ascending for
//! upper, descending for lower), including reference `dtrsm`'s
//! multiply-by-reciprocal on the diagonal. For UPPER the per-element
//! add order stays fully ascending — bit-identical to the plain
//! sweep; for LOWER the out-of-group adds now precede the in-group
//! adds — a deterministic, documented reorder (the tests' scalar
//! replay mirrors it). The fused-FMA variant is DEFERRED at the f64
//! campaign close (relaxed-madd rounding is implementation-dependent
//! — architect decision; DESIGN.md, "Fast mode is a lever"). Transpose forms:
//! planned (STATUS.md, todo S1).

use super::check_mat;
use crate::{Diag, Error, Side, Trans, Uplo};
use crate::kernels::daxpy4;
use crate::L1::{daxpy_k, dscal};
use crate::L2::dtrsv_k;

/// `B ← α·op(A)⁻¹·B` ([`Side::Left`]) or `B ← α·B·op(A)⁻¹` ([`Side::Right`])
/// for a triangular `A`, the triangle named by `uplo` stored, with a
/// unit diagonal when `diag` says so; `A` is `m×m` on the left and `n×n`
/// on the right, `B` is `m×n`, each at its own leading dimension.
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ` per `transa`. `work` is scratch of at
/// least [`dtrsm_work`]`(side, transa, m, n)` elements.
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
pub fn dtrsm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize, b: &mut [f64], ldb: usize, work: &mut [f64]) -> Result<(), Error> {
	const R: &str = "dtrsm";
	if transa != Trans::NoTrans {
		return Err(crate::error::unsupported(R, 3));
	}
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 8, 9, a.len(), na, na, lda)?;
	crate::error::mat(R, 10, 11, b.len(), m, n, ldb)?;
	crate::error::work(R, 12, work.len(), dtrsm_work(side, transa, m, n))?;
	match side {
		Side::Left => dtrsm_left_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb),
		Side::Right => dtrsm_right_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb),
	}
	Ok(())
}

/// Elements of scratch [`dtrsm`] needs for these dimensions. Zero: no form
/// built so far uses scratch.
pub fn dtrsm_work(_side: Side, _transa: Trans, _m: usize, _n: usize) -> usize {
	0
}

/// B ← αA⁻¹B. A is m×m triangular, B is m×n. No singularity check —
/// a zero diagonal yields inf/NaN, as in reference BLAS.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dtrsm_left_k(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f64],
	bcs: usize,
) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let mut j = 0usize;
	while j + 4 <= n {
		if alpha != 1.0 {
			for u in 0..4 {
				dscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		let bp = b.as_mut_ptr();
		let cols: [*mut f64; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// the dtrsv elimination, four right-hand sides in lockstep
		// sharing each A column read
		if upper {
			for l in (0..m).rev() {
				let d = a[l * acs + l];
				let mut t = [0.0f64; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) /= d;
						}
						t[u] = -*cu.add(l);
					}
				}
				unsafe {
					daxpy4(a.as_ptr().add(l * acs), t, cols[0], cols[1], cols[2], cols[3], l);
				}
			}
		} else {
			for l in 0..m {
				let d = a[l * acs + l];
				let mut t = [0.0f64; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) /= d;
						}
						t[u] = -*cu.add(l);
					}
				}
				unsafe {
					daxpy4(
						a.as_ptr().add(l * acs + l + 1),
						t,
						cols[0].add(l + 1),
						cols[1].add(l + 1),
						cols[2].add(l + 1),
						cols[3].add(l + 1),
						m - l - 1,
					);
				}
			}
		}
		j += 4;
	}
	while j < n {
		let col = &mut b[j * bcs..j * bcs + m];
		if alpha != 1.0 {
			dscal(alpha, col);
		}
		dtrsv_k(m, a, acs, upper, unit, col);
		j += 1;
	}
}

/// B ← αBA⁻¹. A is n×n triangular, B is m×n. Reference `dtrsm`
/// multiplies by the diagonal's reciprocal (not a division) — kept.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dtrsm_right_k(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f64],
	bcs: usize,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	// X·A = αB solved column-wise: X[:,j] = (αB[:,j] − Σ X[:,k]·a`[k,j]`)
	// / a`[j,j]`, where the sum runs over already-solved columns k
	// (k < j for upper, k > j for lower). Groups of four apply the
	// solved columns outside the group first (one shared stream each),
	// then finish the in-group elimination in the original order.
	// Columns never alias.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const f64, m) };
	let do_col = |j: usize, lo: usize, hi: usize| {
		if alpha != 1.0 {
			dscal(alpha, col_mut(j));
		}
		for k in lo..hi {
			daxpy_k(-a[j * acs + k], col_ref(k), col_mut(j));
		}
		if !unit {
			dscal(1.0 / a[j * acs + j], col_mut(j));
		}
	};
	let fan_out = |gs: usize, k: usize| {
		let t = [
			-a[gs * acs + k],
			-a[(gs + 1) * acs + k],
			-a[(gs + 2) * acs + k],
			-a[(gs + 3) * acs + k],
		];
		unsafe {
			daxpy4(
				bp.add(k * bcs) as *const f64,
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
		let mut gs = 0usize;
		while gs + 4 <= n {
			if alpha != 1.0 {
				for u in 0..4 {
					dscal(alpha, col_mut(gs + u));
				}
			}
			// solved columns before the group, one shared stream each —
			// ascending k, so the per-element order stays the plain
			// sweep's (bit-identical for upper)
			for k in 0..gs {
				fan_out(gs, k);
			}
			// in-group elimination, original ascending order
			for tc in gs..gs + 4 {
				for k in gs..tc {
					daxpy_k(-a[tc * acs + k], col_ref(k), col_mut(tc));
				}
				if !unit {
					dscal(1.0 / a[tc * acs + tc], col_mut(tc));
				}
			}
			gs += 4;
		}
		for j in gs..n {
			do_col(j, 0, j);
		}
	} else {
		let r = n % 4;
		let mut gs = n;
		while gs >= r + 4 {
			gs -= 4;
			if alpha != 1.0 {
				for u in 0..4 {
					dscal(alpha, col_mut(gs + u));
				}
			}
			// solved columns after the group, one shared stream each
			for k in gs + 4..n {
				fan_out(gs, k);
			}
			// in-group elimination, original descending order
			for tc in (gs..gs + 4).rev() {
				for k in tc + 1..gs + 4 {
					daxpy_k(-a[tc * acs + k], col_ref(k), col_mut(tc));
				}
				if !unit {
					dscal(1.0 / a[tc * acs + tc], col_mut(tc));
				}
			}
		}
		for j in (0..r).rev() {
			do_col(j, j + 1, n);
		}
	}
}
