//! `ctrsm` — triangular solve, many right-hand sides, in place:
//! B ← αA⁻¹B (left) or B ← αBA⁻¹ (right), complex.
//!
//! Implementation: the `dtrsm` shapes verbatim with complex scalars —
//! left side is αscal then four B columns walking the `ctrsv`
//! elimination in lockstep sharing each A column read
//! (`kernels::caxpy4`; divisions by Smith's algorithm, `C32`'s
//! guarded `/`); right side fans already-solved out-of-group columns
//! in via one shared stream each, then finishes the in-group
//! elimination in the original order, multiplying by the diagonal's
//! reciprocal (reference `ctrsm`'s ONE/A(j,j) — kept). Same reorder
//! disclosure as the real layers: UPPER-right stays bit-for-bit the plain sweep;
//! LOWER-right's out-of-group adds precede the in-group adds —
//! deterministic, documented, mirrored by the tests' replay. No
//! singularity check. The transposed and
//! conjugate-transposed forms route between a workspace copy (left) and
//! the no-copy accessor kernel (right) per the race (`bench/README.md`).

use super::super::check_mat;
use crate::{Diag, Error, Side, Trans, Uplo};
use crate::c32::C32;
use crate::kernels::caxpy4;
use crate::L1::{caxpy_k, cscal};
use crate::L2::ctrsv_k;

/// `B ← α·op(A)⁻¹·B` ([`Side::Left`]) or `B ← α·B·op(A)⁻¹` ([`Side::Right`])
/// for a triangular `A`, the triangle named by `uplo` stored, with a
/// unit diagonal when `diag` says so; `A` is `m×m` on the left and `n×n`
/// on the right, `B` is `m×n`, each at its own leading dimension.
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ` per `transa`. `work` is scratch of at
/// least [`ctrsm_work`]`(side, transa, m, n)` elements.
///
/// The transposed forms are defined as the no-transpose form on an
/// explicit transpose (conjugate transpose), and two
/// candidates compute them, [`ctrsm_copy`] and [`ctrsm_stream`],
/// both bit-identical to that definition; which one this routine routes
/// to per side is the race's verdict (`bench/README.md`; decision D3 in `DESIGN.md`).
///
/// Errors: a leading dimension below its row count (arguments 9, 11);
/// `a` or `b` shorter than its extent (arguments 8, 10); `work` shorter
/// than the query (argument 12).
#[allow(clippy::too_many_arguments)]
pub fn ctrsm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: C32, a: &[C32], lda: usize, b: &mut [C32], ldb: usize, work: &mut [C32]) -> Result<(), Error> {
	const R: &str = "ctrsm";
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 8, 9, a.len(), na, na, lda)?;
	crate::error::mat(R, 10, 11, b.len(), m, n, ldb)?;
	crate::error::work(R, 12, work.len(), ctrsm_work(side, transa, m, n))?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if alpha == C32::ZERO {
		// reference: B ← 0 without reading A or B
		for j in 0..n {
			b[j * ldb..j * ldb + m].fill(C32::ZERO);
		}
		return Ok(());
	}
	match (side, transa) {
		(Side::Left, Trans::NoTrans) => ctrsm_left_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb),
		(Side::Right, Trans::NoTrans) => ctrsm_right_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb, Trans::NoTrans),
		(Side::Left, _) if LEFT_T_STREAM => ctrsm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		(Side::Right, _) if RIGHT_T_STREAM => ctrsm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		_ => ctrsm_copy(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, work),
	}
	Ok(())
}

/// Elements of scratch [`ctrsm`] needs for these flags and dimensions:
/// zero for the no-transpose forms and for a transposed form routed to
/// the streaming candidate; the `na×na` copy ([`ctrsm_copy_work`])
/// where the copy candidate is routed.
pub fn ctrsm_work(side: Side, transa: Trans, m: usize, n: usize) -> usize {
	if m == 0 || n == 0 || transa == Trans::NoTrans {
		return 0;
	}
	let stream = if side == Side::Left { LEFT_T_STREAM } else { RIGHT_T_STREAM };
	if stream { 0 } else { ctrsm_copy_work(side, m, n) }
}

// Which candidate the transposed forms route to, per side — the race's
// verdict (two runner draws, `bench/README.md`): on the right the stream
// (transposed accessor) won 1.02–1.6× in every type; on the left the
// scalar complex reduction kernel lost to the copy 0.3–0.75× at every
// size but the skinniest, so the copy is routed here and the lane
// version of that kernel is a recorded lever. Both candidates stay
// exported and tested.
const LEFT_T_STREAM: bool = false;
const RIGHT_T_STREAM: bool = true;

/// B ← αA⁻¹B. A is m×m triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ctrsm_left_k(
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
		if alpha != C32::ONE {
			for u in 0..4 {
				cscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		let bp = b.as_mut_ptr();
		let cols: [*mut C32; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// the ctrsv elimination, four right-hand sides in lockstep
		// sharing each A column read
		if upper {
			for l in (0..m).rev() {
				let d = a[l * acs + l];
				let mut t = [C32::ZERO; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) = *cu.add(l) / d;
						}
						t[u] = -*cu.add(l);
					}
				}
				unsafe {
					caxpy4(a.as_ptr().add(l * acs), t, cols[0], cols[1], cols[2], cols[3], l);
				}
			}
		} else {
			for l in 0..m {
				let d = a[l * acs + l];
				let mut t = [C32::ZERO; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) = *cu.add(l) / d;
						}
						t[u] = -*cu.add(l);
					}
				}
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
			}
		}
		j += 4;
	}
	while j < n {
		let col = &mut b[j * bcs..j * bcs + m];
		if alpha != C32::ONE {
			cscal(alpha, col);
		}
		ctrsv_k(m, a, acs, upper, unit, col);
		j += 1;
	}
}

/// B ← αBA⁻¹. A is n×n triangular, B is m×n. Reference `ctrsm`
/// multiplies by the diagonal's reciprocal (not a division) — kept.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ctrsm_right_k(
	alpha: C32,
	m: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [C32],
	bcs: usize,
	op: Trans,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	// `upper` and the accessor describe op(A): for a transposed op the
	// stored triangle is the other one and the element roles swap
	let at = |row: usize, col: usize| -> C32 {
		match op {
			Trans::NoTrans => a[col * acs + row],
			Trans::Trans => a[row * acs + col],
			Trans::ConjTrans => a[row * acs + col].conj(),
		}
	};
	// X·A = αB solved column-wise: X[:,j] = (αB[:,j] − Σ X[:,k]·a`[k,j]`)
	// / a`[j,j]`, the sum over already-solved columns k (k < j for
	// upper, k > j for lower). Columns never alias.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const C32, m) };
	let do_col = |j: usize, lo: usize, hi: usize| {
		if alpha != C32::ONE {
			cscal(alpha, col_mut(j));
		}
		for k in lo..hi {
			caxpy_k(-at(k, j), col_ref(k), col_mut(j));
		}
		if !unit {
			cscal(C32::ONE / at(j, j), col_mut(j));
		}
	};
	let fan_out = |gs: usize, k: usize| {
		let t = [
			-at(k, gs),
			-at(k, gs + 1),
			-at(k, gs + 2),
			-at(k, gs + 3),
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
		let mut gs = 0usize;
		while gs + 4 <= n {
			if alpha != C32::ONE {
				for u in 0..4 {
					cscal(alpha, col_mut(gs + u));
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
					caxpy_k(-at(k, tc), col_ref(k), col_mut(tc));
				}
				if !unit {
					cscal(C32::ONE / at(tc, tc), col_mut(tc));
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
			if alpha != C32::ONE {
				for u in 0..4 {
					cscal(alpha, col_mut(gs + u));
				}
			}
			// solved columns after the group, one shared stream each
			for k in gs + 4..n {
				fan_out(gs, k);
			}
			// in-group elimination, original descending order
			for tc in (gs..gs + 4).rev() {
				for k in tc + 1..gs + 4 {
					caxpy_k(-at(k, tc), col_ref(k), col_mut(tc));
				}
				if !unit {
					cscal(C32::ONE / at(tc, tc), col_mut(tc));
				}
			}
		}
		for j in (0..r).rev() {
			do_col(j, j + 1, n);
		}
	}
}

/// Candidate for the transposed forms (decision D3 in `DESIGN.md`; the race in `bench/README.md`): op(A)
/// is transposed (and conjugated) into `work` and the no-transpose
/// kernel runs on the copy with the triangle flipped, so the result is
/// the explicit-copy definition itself. `work` needs
/// [`ctrsm_copy_work`]`(side, m, n)` elements. Exported so the
/// tests and the benchmarks can race it; it asserts rather than
/// returning errors, and `transa` must not be `NoTrans`.
#[allow(clippy::too_many_arguments)]
pub fn ctrsm_copy(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: C32, a: &[C32], lda: usize, b: &mut [C32], ldb: usize, work: &mut [C32]) {
	assert!(transa != Trans::NoTrans, "ctrsm_copy: a transposed form only");
	let na = if side == Side::Left { m } else { n };
	check_mat(a.len(), na, na, lda);
	assert!(work.len() >= na * na, "ctrsm_copy: workspace shorter than ctrsm_copy_work");
	let ta = &mut work[..na * na];
		if transa == Trans::ConjTrans {
			super::super::transpose_into(a, lda, na, na, ta, |x| x.conj());
		} else {
			super::super::transpose_into(a, lda, na, na, ta, |x| x);
		}
	let upper = !uplo.is_upper();
	match side {
		Side::Left => ctrsm_left_k(alpha, m, n, ta, na.max(1), upper, diag.is_unit(), b, ldb),
		Side::Right => ctrsm_right_k(alpha, m, n, ta, na.max(1), upper, diag.is_unit(), b, ldb, Trans::NoTrans),
	}
}

/// Elements of scratch [`ctrsm_copy`] needs: the `na×na` copy of
/// op(A), `na` being `m` on the left and `n` on the right.
pub fn ctrsm_copy_work(side: Side, m: usize, n: usize) -> usize {
	let na = if side == Side::Left { m } else { n };
	na * na
}

/// Candidate for the transposed forms (decision D3 in `DESIGN.md`; the race in `bench/README.md`): no
/// copy; on the right the no-transpose kernel reads A through a
/// transposed accessor (it only ever reads A as scalars), on the left
/// `ctrsm_left_t_k` reduces down the stored columns. Bit-identical
/// to [`ctrsm_copy`] (tested). Exported so the tests and the
/// benchmarks can race it; it asserts rather than returning errors, and
/// `transa` must not be `NoTrans`.
#[allow(clippy::too_many_arguments)]
pub fn ctrsm_stream(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: C32, a: &[C32], lda: usize, b: &mut [C32], ldb: usize) {
	assert!(transa != Trans::NoTrans, "ctrsm_stream: a transposed form only");
	let upper = !uplo.is_upper();
	match side {
		Side::Left => ctrsm_left_t_k(alpha, m, n, a, lda, upper, transa == Trans::ConjTrans, diag.is_unit(), b, ldb),
		Side::Right => ctrsm_right_k(alpha, m, n, a, lda, upper, diag.is_unit(), b, ldb, transa),
	}
}

/// B ← α·op(A)⁻¹·B with op(A) = Aᵀ (`conj`: Aᴴ), computed from the
/// stored A without a copy: `upper` names op(A)'s triangle (stored A
/// holds the other). Each element's arithmetic sequence is the lockstep
/// ctrsv elimination's — αb first, then the products subtracted one at a time down the
/// stored column (descending for upper, ascending for lower), then the
/// division — in the scalar complex product form, which is
/// bit-identical to the lane form (`c32.rs`), so the result equals
/// [`ctrsm_left_k`] on an explicit transpose (tested). Four right-hand
/// sides go in lockstep and rows two at a time, so one step loads two A
/// values and four B values for eight complex multiply-adds; op(A)'s
/// row l is stored column l, read contiguously. Conjugation is applied
/// to each A value as it is read (an exact sign flip).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ctrsm_left_t_k(
	alpha: C32,
	m: usize,
	n: usize,
	a: &[C32],
	acs: usize,
	upper: bool,
	conj: bool,
	unit: bool,
	b: &mut [C32],
	bcs: usize,
) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let cj = |v: C32| if conj { v.conj() } else { v };
	let ap = a.as_ptr();
	let mut j = 0usize;
	while j + 4 <= n {
		if alpha != C32::ONE {
			for u in 0..4 {
				cscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		let bp = b.as_mut_ptr();
		let c: [*mut C32; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// SAFETY: the four columns are distinct columns of `b`, each of
		// `m` elements, and `a` is read only through `ap` within its
		// checked extent
		unsafe {
			if upper {
				let r = m % 2;
				let mut lb = m;
				while lb >= r + 2 {
					lb -= 2;
					let mut acc: [[C32; 4]; 2] = [[*c[0].add(lb), *c[1].add(lb), *c[2].add(lb), *c[3].add(lb)], [*c[0].add(lb + 1), *c[1].add(lb + 1), *c[2].add(lb + 1), *c[3].add(lb + 1)]];
					let mut i = m;
					while i > lb + 2 {
						i -= 1;
						let a0 = cj(*ap.add(lb * acs + i));
						let a1 = cj(*ap.add((lb + 1) * acs + i));
						for u in 0..4 {
							let t = -*c[u].add(i);
							acc[0][u] = acc[0][u] + a0 * t;
							acc[1][u] = acc[1][u] + a1 * t;
						}
					}
					if !unit {
						let d = cj(*ap.add((lb + 1) * acs + lb + 1));
						for v in acc[1].iter_mut() {
							*v = *v / d;
						}
					}
					let a01 = cj(*ap.add(lb * acs + lb + 1));
					for u in 0..4 {
						*c[u].add(lb + 1) = acc[1][u];
						acc[0][u] = acc[0][u] + a01 * -acc[1][u];
					}
					if !unit {
						let d = cj(*ap.add(lb * acs + lb));
						for v in acc[0].iter_mut() {
							*v = *v / d;
						}
					}
					for u in 0..4 {
						*c[u].add(lb) = acc[0][u];
					}
				}
				if r == 1 {
					let l = 0usize;
					let mut acc = [*c[0].add(l), *c[1].add(l), *c[2].add(l), *c[3].add(l)];
					let mut i = m;
					while i > l + 1 {
						i -= 1;
						let av = cj(*ap.add(l * acs + i));
						for (u, v) in acc.iter_mut().enumerate() {
							*v = *v + av * -*c[u].add(i);
						}
					}
					if !unit {
						let d = cj(*ap.add(l * acs + l));
						for v in acc.iter_mut() {
							*v = *v / d;
						}
					}
					for (u, v) in acc.iter().enumerate() {
						*c[u].add(l) = *v;
					}
				}
			} else {
				let mut lb = 0usize;
				while lb + 2 <= m {
					let mut acc: [[C32; 4]; 2] = [[*c[0].add(lb), *c[1].add(lb), *c[2].add(lb), *c[3].add(lb)], [*c[0].add(lb + 1), *c[1].add(lb + 1), *c[2].add(lb + 1), *c[3].add(lb + 1)]];
					for i in 0..lb {
						let a0 = cj(*ap.add(lb * acs + i));
						let a1 = cj(*ap.add((lb + 1) * acs + i));
						for u in 0..4 {
							let t = -*c[u].add(i);
							acc[0][u] = acc[0][u] + a0 * t;
							acc[1][u] = acc[1][u] + a1 * t;
						}
					}
					if !unit {
						let d = cj(*ap.add(lb * acs + lb));
						for v in acc[0].iter_mut() {
							*v = *v / d;
						}
					}
					let a10 = cj(*ap.add((lb + 1) * acs + lb));
					for u in 0..4 {
						*c[u].add(lb) = acc[0][u];
						acc[1][u] = acc[1][u] + a10 * -acc[0][u];
					}
					if !unit {
						let d = cj(*ap.add((lb + 1) * acs + lb + 1));
						for v in acc[1].iter_mut() {
							*v = *v / d;
						}
					}
					for u in 0..4 {
						*c[u].add(lb + 1) = acc[1][u];
					}
					lb += 2;
				}
				if lb < m {
					let l = lb;
					let mut acc = [*c[0].add(l), *c[1].add(l), *c[2].add(l), *c[3].add(l)];
					for i in 0..l {
						let av = cj(*ap.add(l * acs + i));
						for (u, v) in acc.iter_mut().enumerate() {
							*v = *v + av * -*c[u].add(i);
						}
					}
					if !unit {
						let d = cj(*ap.add(l * acs + l));
						for v in acc.iter_mut() {
							*v = *v / d;
						}
					}
					for (u, v) in acc.iter().enumerate() {
						*c[u].add(l) = *v;
					}
				}
			}
		}
		j += 4;
	}
	while j < n {
		let x = &mut b[j * bcs..j * bcs + m];
		if alpha != C32::ONE {
			cscal(alpha, x);
		}
		if upper {
			let mut l = m;
		while l > 0 {
			l -= 1;
				let mut acc = x[l];
				let mut i = m;
			while i > l + 1 {
				i -= 1;
				acc = acc + cj(a[l * acs + i]) * -x[i];
			}
				if !unit {
				acc = acc / cj(a[l * acs + l]);
			}
				x[l] = acc;
			}
		} else {
			for l in 0..m {
				let mut acc = x[l];
				for i in 0..l {
				acc = acc + cj(a[l * acs + i]) * -x[i];
			}
				if !unit {
				acc = acc / cj(a[l * acs + l]);
			}
				x[l] = acc;
			}
		}
		j += 1;
	}
}
