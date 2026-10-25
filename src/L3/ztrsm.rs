//! `ztrsm` — triangular solve, many right-hand sides, in place:
//! B ← αA⁻¹B (left) or B ← αBA⁻¹ (right), complex.
//!
//! Implementation: the `dtrsm` shapes verbatim with complex scalars —
//! left side is αscal then four B columns walking the `ztrsv`
//! elimination in lockstep sharing each A column read
//! (`kernels::zaxpy4`; divisions by Smith's algorithm, `C64`'s
//! guarded `/`); right side fans already-solved out-of-group columns
//! in via one shared stream each, then finishes the in-group
//! elimination in the original order, multiplying by the diagonal's
//! reciprocal (reference `ztrsm`'s ONE/A(j,j) — kept). Same reorder
//! disclosure as f64: UPPER-right stays bit-for-bit the plain sweep;
//! LOWER-right's out-of-group adds precede the in-group adds —
//! deterministic, documented, mirrored by the tests' replay. No
//! singularity check. Transpose/conjugate forms: not built — no
//! consumer yet.

use super::check_mat;
use crate::{Diag, Error, Side, Trans, Uplo};
use crate::c64::C64;
use crate::kernels::zaxpy4;
use crate::L1::{zaxpy_k, zscal};
use crate::L2::ztrsv_k;

/// `B ← α·op(A)⁻¹·B` ([`Side::Left`]) or `B ← α·B·op(A)⁻¹` ([`Side::Right`])
/// for a triangular `A`, the triangle named by `uplo` stored, with a
/// unit diagonal when `diag` says so; `A` is `m×m` on the left and `n×n`
/// on the right, `B` is `m×n`, each at its own leading dimension.
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ` per `transa`. `work` is scratch of at
/// least [`ztrsm_work`]`(side, transa, m, n)` elements.
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
pub fn ztrsm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: C64, a: &[C64], lda: usize, b: &mut [C64], ldb: usize, work: &mut [C64]) -> Result<(), Error> {
	const R: &str = "ztrsm";
	if transa != Trans::NoTrans {
		return Err(crate::error::unsupported(R, 3));
	}
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 8, 9, a.len(), na, na, lda)?;
	crate::error::mat(R, 10, 11, b.len(), m, n, ldb)?;
	crate::error::work(R, 12, work.len(), ztrsm_work(side, transa, m, n))?;
	match side {
		Side::Left => ztrsm_left_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb),
		Side::Right => ztrsm_right_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb),
	}
	Ok(())
}

/// Elements of scratch [`ztrsm`] needs for these dimensions. Zero: no form
/// built so far uses scratch.
pub fn ztrsm_work(_side: Side, _transa: Trans, _m: usize, _n: usize) -> usize {
	0
}

/// B ← αA⁻¹B. A is m×m triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ztrsm_left_k(
	alpha: C64,
	m: usize,
	n: usize,
	a: &[C64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [C64],
	bcs: usize,
) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let mut j = 0usize;
	while j + 4 <= n {
		if alpha != C64::ONE {
			for u in 0..4 {
				zscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		let bp = b.as_mut_ptr();
		let cols: [*mut C64; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// the ztrsv elimination, four right-hand sides in lockstep
		// sharing each A column read
		if upper {
			for l in (0..m).rev() {
				let d = a[l * acs + l];
				let mut t = [C64::ZERO; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) = *cu.add(l) / d;
						}
						t[u] = -*cu.add(l);
					}
				}
				unsafe {
					zaxpy4(a.as_ptr().add(l * acs), t, cols[0], cols[1], cols[2], cols[3], l);
				}
			}
		} else {
			for l in 0..m {
				let d = a[l * acs + l];
				let mut t = [C64::ZERO; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) = *cu.add(l) / d;
						}
						t[u] = -*cu.add(l);
					}
				}
				unsafe {
					zaxpy4(
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
		if alpha != C64::ONE {
			zscal(alpha, col);
		}
		ztrsv_k(m, a, acs, upper, unit, col);
		j += 1;
	}
}

/// B ← αBA⁻¹. A is n×n triangular, B is m×n. Reference `ztrsm`
/// multiplies by the diagonal's reciprocal (not a division) — kept.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ztrsm_right_k(
	alpha: C64,
	m: usize,
	n: usize,
	a: &[C64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [C64],
	bcs: usize,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	// X·A = αB solved column-wise: X[:,j] = (αB[:,j] − Σ X[:,k]·a`[k,j]`)
	// / a`[j,j]`, the sum over already-solved columns k (k < j for
	// upper, k > j for lower). Columns never alias.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const C64, m) };
	let do_col = |j: usize, lo: usize, hi: usize| {
		if alpha != C64::ONE {
			zscal(alpha, col_mut(j));
		}
		for k in lo..hi {
			zaxpy_k(-a[j * acs + k], col_ref(k), col_mut(j));
		}
		if !unit {
			zscal(C64::ONE / a[j * acs + j], col_mut(j));
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
			zaxpy4(
				bp.add(k * bcs) as *const C64,
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
			if alpha != C64::ONE {
				for u in 0..4 {
					zscal(alpha, col_mut(gs + u));
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
					zaxpy_k(-a[tc * acs + k], col_ref(k), col_mut(tc));
				}
				if !unit {
					zscal(C64::ONE / a[tc * acs + tc], col_mut(tc));
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
			if alpha != C64::ONE {
				for u in 0..4 {
					zscal(alpha, col_mut(gs + u));
				}
			}
			// solved columns after the group, one shared stream each
			for k in gs + 4..n {
				fan_out(gs, k);
			}
			// in-group elimination, original descending order
			for tc in (gs..gs + 4).rev() {
				for k in tc + 1..gs + 4 {
					zaxpy_k(-a[tc * acs + k], col_ref(k), col_mut(tc));
				}
				if !unit {
					zscal(C64::ONE / a[tc * acs + tc], col_mut(tc));
				}
			}
		}
		for j in (0..r).rev() {
			do_col(j, j + 1, n);
		}
	}
}
