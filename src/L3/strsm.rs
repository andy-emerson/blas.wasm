//! `strsm` — triangular solve, many right-hand sides, in place:
//! B ← αA⁻¹B (left) or B ← αBA⁻¹ (right).
//!
//! Implementation: left side is divide-then-
//! column-saxpy — αscal, then four B columns walk the `strsv`
//! elimination in lockstep so each A column streams once per group of
//! four (A traffic 4× down; the columns are independent, so results
//! are bit-for-bit the plain strsv-per-column loop, tested). Right
//! side groups four destination columns: already-solved columns
//! outside the group fan out via one shared stream each, then the
//! in-group elimination runs in the original order (ascending for
//! upper, descending for lower), including reference `dtrsm`'s
//! multiply-by-reciprocal on the diagonal. For UPPER the per-element
//! add order stays fully ascending — bit-identical to the plain
//! sweep; for LOWER the out-of-group adds now precede the in-group
//! adds — a deterministic, documented reorder (the tests' scalar
//! replay mirrors it). The fused-FMA variant is DEFERRED at the f32
//! campaign close (relaxed-madd rounding is implementation-dependent
//! — architect decision; DESIGN.md, "Fast mode is a lever"). The transposed forms route
//! between a workspace copy and the no-copy kernels per the race
//! (`bench/README.md`).

use super::check_mat;
use crate::{Diag, Error, Side, Trans, Uplo};
use crate::kernels::saxpy4;
use crate::L1::{saxpy_k, sscal};
use crate::L2::strsv_k;

/// `B ← α·op(A)⁻¹·B` ([`Side::Left`]) or `B ← α·B·op(A)⁻¹` ([`Side::Right`])
/// for a triangular `A`, the triangle named by `uplo` stored, with a
/// unit diagonal when `diag` says so; `A` is `m×m` on the left and `n×n`
/// on the right, `B` is `m×n`, each at its own leading dimension.
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ` per `transa`. `work` is scratch of at
/// least [`strsm_work`]`(side, transa, m, n)` elements.
///
/// The transposed forms are defined as the no-transpose form on an
/// explicit transpose; `ConjTrans` is `Trans` for a real matrix, and two
/// candidates compute them, [`strsm_copy`] and [`strsm_stream`],
/// both bit-identical to that definition; which one this routine routes
/// to per side is the race's verdict (`STATUS.md`, D3).
///
/// Errors: a leading dimension below its row count (arguments 9, 11);
/// `a` or `b` shorter than its extent (arguments 8, 10); `work` shorter
/// than the query (argument 12).
#[allow(clippy::too_many_arguments)]
pub fn strsm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f32, a: &[f32], lda: usize, b: &mut [f32], ldb: usize, work: &mut [f32]) -> Result<(), Error> {
	const R: &str = "strsm";
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 8, 9, a.len(), na, na, lda)?;
	crate::error::mat(R, 10, 11, b.len(), m, n, ldb)?;
	crate::error::work(R, 12, work.len(), strsm_work(side, transa, m, n))?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if alpha == 0.0 {
		// reference: B ← 0 without reading A or B
		for j in 0..n {
			b[j * ldb..j * ldb + m].fill(0.0);
		}
		return Ok(());
	}
	match (side, transa) {
		(Side::Left, Trans::NoTrans) => strsm_left_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb),
		(Side::Right, Trans::NoTrans) => strsm_right_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb, Trans::NoTrans),
		(Side::Left, _) if LEFT_T_STREAM => strsm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		(Side::Right, _) if RIGHT_T_STREAM => strsm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		_ => strsm_copy(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, work),
	}
	Ok(())
}

/// Elements of scratch [`strsm`] needs for these flags and dimensions:
/// zero for the no-transpose forms and for a transposed form routed to
/// the streaming candidate; the `na×na` copy ([`strsm_copy_work`])
/// where the copy candidate is routed.
pub fn strsm_work(side: Side, transa: Trans, m: usize, n: usize) -> usize {
	if m == 0 || n == 0 || transa == Trans::NoTrans {
		return 0;
	}
	let stream = if side == Side::Left { LEFT_T_STREAM } else { RIGHT_T_STREAM };
	if stream { 0 } else { strsm_copy_work(side, m, n) }
}

// Which candidate the transposed forms route to, per side — the race's
// verdict (two runner draws, `bench/README.md`): the stream won both
// sides in the real types (left 1.3–2.8× for f64 at every size from 256,
// f32 1.0–2.4× with the ascending-walk cases at 0.85–1.0 on several
// square shapes; right 1.02–1.6×), needing no workspace. Both candidates
// stay exported and tested.
const LEFT_T_STREAM: bool = true;
const RIGHT_T_STREAM: bool = true;

/// B ← αA⁻¹B. A is m×m triangular, B is m×n. No singularity check —
/// a zero diagonal yields inf/NaN, as in reference BLAS.
#[allow(clippy::too_many_arguments)]
pub(crate) fn strsm_left_k(
	alpha: f32,
	m: usize,
	n: usize,
	a: &[f32],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f32],
	bcs: usize,
) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let mut j = 0usize;
	while j + 4 <= n {
		if alpha != 1.0 {
			for u in 0..4 {
				sscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		let bp = b.as_mut_ptr();
		let cols: [*mut f32; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// the strsv elimination, four right-hand sides in lockstep
		// sharing each A column read
		if upper {
			for l in (0..m).rev() {
				let d = a[l * acs + l];
				let mut t = [0.0f32; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) /= d;
						}
						t[u] = -*cu.add(l);
					}
				}
				unsafe {
					saxpy4(a.as_ptr().add(l * acs), t, cols[0], cols[1], cols[2], cols[3], l);
				}
			}
		} else {
			for l in 0..m {
				let d = a[l * acs + l];
				let mut t = [0.0f32; 4];
				for (u, cu) in cols.iter().enumerate() {
					unsafe {
						if !unit {
							*cu.add(l) /= d;
						}
						t[u] = -*cu.add(l);
					}
				}
				unsafe {
					saxpy4(
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
			sscal(alpha, col);
		}
		strsv_k(m, a, acs, upper, unit, col);
		j += 1;
	}
}

/// B ← αBA⁻¹. A is n×n triangular, B is m×n. Reference `dtrsm`
/// multiplies by the diagonal's reciprocal (not a division) — kept.
#[allow(clippy::too_many_arguments)]
pub(crate) fn strsm_right_k(
	alpha: f32,
	m: usize,
	n: usize,
	a: &[f32],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f32],
	bcs: usize,
	op: Trans,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	// `upper` and the accessor describe op(A): for a transposed op the
	// stored triangle is the other one and the element roles swap
	let at = |row: usize, col: usize| -> f32 {
		if op == Trans::NoTrans {
			a[col * acs + row]
		} else {
			a[row * acs + col]
		}
	};
	// X·A = αB solved column-wise: X[:,j] = (αB[:,j] − Σ X[:,k]·a`[k,j]`)
	// / a`[j,j]`, where the sum runs over already-solved columns k
	// (k < j for upper, k > j for lower). Groups of four apply the
	// solved columns outside the group first (one shared stream each),
	// then finish the in-group elimination in the original order.
	// Columns never alias.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const f32, m) };
	let do_col = |j: usize, lo: usize, hi: usize| {
		if alpha != 1.0 {
			sscal(alpha, col_mut(j));
		}
		for k in lo..hi {
			saxpy_k(-at(k, j), col_ref(k), col_mut(j));
		}
		if !unit {
			sscal(1.0 / at(j, j), col_mut(j));
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
			saxpy4(
				bp.add(k * bcs) as *const f32,
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
					sscal(alpha, col_mut(gs + u));
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
					saxpy_k(-at(k, tc), col_ref(k), col_mut(tc));
				}
				if !unit {
					sscal(1.0 / at(tc, tc), col_mut(tc));
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
					sscal(alpha, col_mut(gs + u));
				}
			}
			// solved columns after the group, one shared stream each
			for k in gs + 4..n {
				fan_out(gs, k);
			}
			// in-group elimination, original descending order
			for tc in (gs..gs + 4).rev() {
				for k in tc + 1..gs + 4 {
					saxpy_k(-at(k, tc), col_ref(k), col_mut(tc));
				}
				if !unit {
					sscal(1.0 / at(tc, tc), col_mut(tc));
				}
			}
		}
		for j in (0..r).rev() {
			do_col(j, j + 1, n);
		}
	}
}

/// Candidate for the transposed forms (`STATUS.md`, decision D3): op(A)
/// is transposed into `work` and the no-transpose
/// kernel runs on the copy with the triangle flipped, so the result is
/// the explicit-copy definition itself. `work` needs
/// [`strsm_copy_work`]`(side, m, n)` elements. Exported so the
/// tests and the benchmarks can race it; it asserts rather than
/// returning errors, and `transa` must not be `NoTrans`.
#[allow(clippy::too_many_arguments)]
pub fn strsm_copy(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f32, a: &[f32], lda: usize, b: &mut [f32], ldb: usize, work: &mut [f32]) {
	assert!(transa != Trans::NoTrans, "strsm_copy: a transposed form only");
	let na = if side == Side::Left { m } else { n };
	check_mat(a.len(), na, na, lda);
	assert!(work.len() >= na * na, "strsm_copy: workspace shorter than strsm_copy_work");
	let ta = &mut work[..na * na];
		super::transpose_into(a, lda, na, na, ta, |x| x);
	let upper = !uplo.is_upper();
	match side {
		Side::Left => strsm_left_k(alpha, m, n, ta, na.max(1), upper, diag.is_unit(), b, ldb),
		Side::Right => strsm_right_k(alpha, m, n, ta, na.max(1), upper, diag.is_unit(), b, ldb, Trans::NoTrans),
	}
}

/// Elements of scratch [`strsm_copy`] needs: the `na×na` copy of
/// op(A), `na` being `m` on the left and `n` on the right.
pub fn strsm_copy_work(side: Side, m: usize, n: usize) -> usize {
	let na = if side == Side::Left { m } else { n };
	na * na
}

/// Candidate for the transposed forms (`STATUS.md`, decision D3): no
/// copy; on the right the no-transpose kernel reads A through a
/// transposed accessor (it only ever reads A as scalars), on the left
/// `strsm_left_t_k` reduces down the stored columns. Bit-identical
/// to [`strsm_copy`] (tested). Exported so the tests and the
/// benchmarks can race it; it asserts rather than returning errors, and
/// `transa` must not be `NoTrans`.
#[allow(clippy::too_many_arguments)]
pub fn strsm_stream(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f32, a: &[f32], lda: usize, b: &mut [f32], ldb: usize) {
	assert!(transa != Trans::NoTrans, "strsm_stream: a transposed form only");
	let upper = !uplo.is_upper();
	match side {
		Side::Left => strsm_left_t_k(alpha, m, n, a, lda, upper, diag.is_unit(), b, ldb),
		Side::Right => strsm_right_k(alpha, m, n, a, lda, upper, diag.is_unit(), b, ldb, transa),
	}
}

/// B ← α·op(A)⁻¹·B with op(A) = Aᵀ, computed from the stored A
/// without a copy: `upper` names op(A)'s triangle (stored A holds the
/// other). Each element's arithmetic sequence is the lockstep
/// strsv elimination's — αb first, then the products subtracted one at a time down the
/// stored column (descending for upper, ascending for lower), then the
/// division — so the result is bit-identical to
/// [`strsm_left_k`] on an explicit transpose (tested). The four
/// right-hand sides of a group ride the lanes, and rows go four at a
/// time so one step loads four A values and four B values for sixteen
/// multiply-adds; op(A)'s row l is stored column l, read contiguously.
#[allow(clippy::too_many_arguments)]
pub(crate) fn strsm_left_t_k(
	alpha: f32,
	m: usize,
	n: usize,
	a: &[f32],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f32],
	bcs: usize,
) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let mut j = 0usize;
	while j + 4 <= n {
		if alpha != 1.0 {
			for u in 0..4 {
				sscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		let bp = b.as_mut_ptr();
		let c: [*mut f32; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		unsafe { strsm_left_t4(m, a, acs, upper, unit, c) };
		j += 4;
	}
	while j < n {
		let col = &mut b[j * bcs..j * bcs + m];
		if alpha != 1.0 {
			sscal(alpha, col);
		}
		strsm_left_t1(m, a, acs, upper, unit, col);
		j += 1;
	}
}

/// Four right-hand sides of [`strsm_left_t_k`] in lockstep.
/// # Safety
/// Each column pointer must be valid for `m` elements; the columns must
/// not alias each other.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn strsm_left_t4(m: usize, a: &[f32], acs: usize, upper: bool, unit: bool, c: [*mut f32; 4]) {
	use crate::lanes::F32x4;
	let ap = a.as_ptr();
	if upper {
		let r = m % 4;
		let mut lb = m;
		while lb >= r + 4 {
			lb -= 4;
			let mut acc: [[F32x4; 1]; 4] = [[F32x4::quad(*c[0].add(lb), *c[1].add(lb), *c[2].add(lb), *c[3].add(lb))], [F32x4::quad(*c[0].add(lb + 1), *c[1].add(lb + 1), *c[2].add(lb + 1), *c[3].add(lb + 1))], [F32x4::quad(*c[0].add(lb + 2), *c[1].add(lb + 2), *c[2].add(lb + 2), *c[3].add(lb + 2))], [F32x4::quad(*c[0].add(lb + 3), *c[1].add(lb + 3), *c[2].add(lb + 3), *c[3].add(lb + 3))]];
			let mut i = m;
			while i > lb + 4 {
				i -= 1;
				let t0 = F32x4::quad(-*c[0].add(i), -*c[1].add(i), -*c[2].add(i), -*c[3].add(i));
				for (v, row) in acc.iter_mut().enumerate() {
					let av = F32x4::splat(*ap.add((lb + v) * acs + i));
					row[0] = row[0].add(av.mul(t0));
				}
			}
				if !unit {
					let d = F32x4::splat(*ap.add((lb + 3) * acs + lb + 3));
					acc[3][0] = acc[3][0].div(d);
				}
				*c[0].add(lb + 3) = acc[3][0].lane0();
				*c[1].add(lb + 3) = acc[3][0].lane1();
				*c[2].add(lb + 3) = acc[3][0].lane2();
				*c[3].add(lb + 3) = acc[3][0].lane3();
				let t0 = F32x4::quad(-acc[3][0].lane0(), -acc[3][0].lane1(), -acc[3][0].lane2(), -acc[3][0].lane3());
				let av = F32x4::splat(*ap.add((lb) * acs + lb + 3));
				acc[0][0] = acc[0][0].add(av.mul(t0));
				let av = F32x4::splat(*ap.add((lb + 1) * acs + lb + 3));
				acc[1][0] = acc[1][0].add(av.mul(t0));
				let av = F32x4::splat(*ap.add((lb + 2) * acs + lb + 3));
				acc[2][0] = acc[2][0].add(av.mul(t0));
				if !unit {
					let d = F32x4::splat(*ap.add((lb + 2) * acs + lb + 2));
					acc[2][0] = acc[2][0].div(d);
				}
				*c[0].add(lb + 2) = acc[2][0].lane0();
				*c[1].add(lb + 2) = acc[2][0].lane1();
				*c[2].add(lb + 2) = acc[2][0].lane2();
				*c[3].add(lb + 2) = acc[2][0].lane3();
				let t0 = F32x4::quad(-acc[2][0].lane0(), -acc[2][0].lane1(), -acc[2][0].lane2(), -acc[2][0].lane3());
				let av = F32x4::splat(*ap.add((lb) * acs + lb + 2));
				acc[0][0] = acc[0][0].add(av.mul(t0));
				let av = F32x4::splat(*ap.add((lb + 1) * acs + lb + 2));
				acc[1][0] = acc[1][0].add(av.mul(t0));
				if !unit {
					let d = F32x4::splat(*ap.add((lb + 1) * acs + lb + 1));
					acc[1][0] = acc[1][0].div(d);
				}
				*c[0].add(lb + 1) = acc[1][0].lane0();
				*c[1].add(lb + 1) = acc[1][0].lane1();
				*c[2].add(lb + 1) = acc[1][0].lane2();
				*c[3].add(lb + 1) = acc[1][0].lane3();
				let t0 = F32x4::quad(-acc[1][0].lane0(), -acc[1][0].lane1(), -acc[1][0].lane2(), -acc[1][0].lane3());
				let av = F32x4::splat(*ap.add((lb) * acs + lb + 1));
				acc[0][0] = acc[0][0].add(av.mul(t0));
				if !unit {
					let d = F32x4::splat(*ap.add((lb) * acs + lb));
					acc[0][0] = acc[0][0].div(d);
				}
				*c[0].add(lb) = acc[0][0].lane0();
				*c[1].add(lb) = acc[0][0].lane1();
				*c[2].add(lb) = acc[0][0].lane2();
				*c[3].add(lb) = acc[0][0].lane3();
		}
		for l in (0..r).rev() {
			let mut acc: [F32x4; 1] = [F32x4::quad(*c[0].add(l), *c[1].add(l), *c[2].add(l), *c[3].add(l))];
			let mut i = m;
			while i > l + 1 {
				i -= 1;
				let t0 = F32x4::quad(-*c[0].add(i), -*c[1].add(i), -*c[2].add(i), -*c[3].add(i));
				let av = F32x4::splat(*ap.add(l * acs + i));
				acc[0] = acc[0].add(av.mul(t0));
			}
			if !unit {
				let d = F32x4::splat(*ap.add(l * acs + l));
				acc[0] = acc[0].div(d);
			}
			*c[0].add(l) = acc[0].lane0();
			*c[1].add(l) = acc[0].lane1();
			*c[2].add(l) = acc[0].lane2();
			*c[3].add(l) = acc[0].lane3();
		}
	} else {
		let mut lb = 0usize;
		while lb + 4 <= m {
			let mut acc: [[F32x4; 1]; 4] = [[F32x4::quad(*c[0].add(lb), *c[1].add(lb), *c[2].add(lb), *c[3].add(lb))], [F32x4::quad(*c[0].add(lb + 1), *c[1].add(lb + 1), *c[2].add(lb + 1), *c[3].add(lb + 1))], [F32x4::quad(*c[0].add(lb + 2), *c[1].add(lb + 2), *c[2].add(lb + 2), *c[3].add(lb + 2))], [F32x4::quad(*c[0].add(lb + 3), *c[1].add(lb + 3), *c[2].add(lb + 3), *c[3].add(lb + 3))]];
			for i in 0..lb {
				let t0 = F32x4::quad(-*c[0].add(i), -*c[1].add(i), -*c[2].add(i), -*c[3].add(i));
				for (v, row) in acc.iter_mut().enumerate() {
					let av = F32x4::splat(*ap.add((lb + v) * acs + i));
					row[0] = row[0].add(av.mul(t0));
				}
			}
				if !unit {
					let d = F32x4::splat(*ap.add((lb) * acs + lb));
					acc[0][0] = acc[0][0].div(d);
				}
				*c[0].add(lb) = acc[0][0].lane0();
				*c[1].add(lb) = acc[0][0].lane1();
				*c[2].add(lb) = acc[0][0].lane2();
				*c[3].add(lb) = acc[0][0].lane3();
				let t0 = F32x4::quad(-acc[0][0].lane0(), -acc[0][0].lane1(), -acc[0][0].lane2(), -acc[0][0].lane3());
				let av = F32x4::splat(*ap.add((lb + 1) * acs + lb));
				acc[1][0] = acc[1][0].add(av.mul(t0));
				let av = F32x4::splat(*ap.add((lb + 2) * acs + lb));
				acc[2][0] = acc[2][0].add(av.mul(t0));
				let av = F32x4::splat(*ap.add((lb + 3) * acs + lb));
				acc[3][0] = acc[3][0].add(av.mul(t0));
				if !unit {
					let d = F32x4::splat(*ap.add((lb + 1) * acs + lb + 1));
					acc[1][0] = acc[1][0].div(d);
				}
				*c[0].add(lb + 1) = acc[1][0].lane0();
				*c[1].add(lb + 1) = acc[1][0].lane1();
				*c[2].add(lb + 1) = acc[1][0].lane2();
				*c[3].add(lb + 1) = acc[1][0].lane3();
				let t0 = F32x4::quad(-acc[1][0].lane0(), -acc[1][0].lane1(), -acc[1][0].lane2(), -acc[1][0].lane3());
				let av = F32x4::splat(*ap.add((lb + 2) * acs + lb + 1));
				acc[2][0] = acc[2][0].add(av.mul(t0));
				let av = F32x4::splat(*ap.add((lb + 3) * acs + lb + 1));
				acc[3][0] = acc[3][0].add(av.mul(t0));
				if !unit {
					let d = F32x4::splat(*ap.add((lb + 2) * acs + lb + 2));
					acc[2][0] = acc[2][0].div(d);
				}
				*c[0].add(lb + 2) = acc[2][0].lane0();
				*c[1].add(lb + 2) = acc[2][0].lane1();
				*c[2].add(lb + 2) = acc[2][0].lane2();
				*c[3].add(lb + 2) = acc[2][0].lane3();
				let t0 = F32x4::quad(-acc[2][0].lane0(), -acc[2][0].lane1(), -acc[2][0].lane2(), -acc[2][0].lane3());
				let av = F32x4::splat(*ap.add((lb + 3) * acs + lb + 2));
				acc[3][0] = acc[3][0].add(av.mul(t0));
				if !unit {
					let d = F32x4::splat(*ap.add((lb + 3) * acs + lb + 3));
					acc[3][0] = acc[3][0].div(d);
				}
				*c[0].add(lb + 3) = acc[3][0].lane0();
				*c[1].add(lb + 3) = acc[3][0].lane1();
				*c[2].add(lb + 3) = acc[3][0].lane2();
				*c[3].add(lb + 3) = acc[3][0].lane3();
			lb += 4;
		}
		for l in lb..m {
			let mut acc: [F32x4; 1] = [F32x4::quad(*c[0].add(l), *c[1].add(l), *c[2].add(l), *c[3].add(l))];
			for i in 0..l {
				let t0 = F32x4::quad(-*c[0].add(i), -*c[1].add(i), -*c[2].add(i), -*c[3].add(i));
				let av = F32x4::splat(*ap.add(l * acs + i));
				acc[0] = acc[0].add(av.mul(t0));
			}
			if !unit {
				let d = F32x4::splat(*ap.add(l * acs + l));
				acc[0] = acc[0].div(d);
			}
			*c[0].add(l) = acc[0].lane0();
			*c[1].add(l) = acc[0].lane1();
			*c[2].add(l) = acc[0].lane2();
			*c[3].add(l) = acc[0].lane3();
		}
	}
}

/// One right-hand side of [`strsm_left_t_k`], scalar, the same
/// per-element sequence.
fn strsm_left_t1(m: usize, a: &[f32], acs: usize, upper: bool, unit: bool, x: &mut [f32]) {
	if upper {
		let mut l = m;
		while l > 0 {
			l -= 1;
			let mut acc = x[l];
			let mut i = m;
			while i > l + 1 {
				i -= 1;
				acc += a[l * acs + i] * -x[i];
			}
			if !unit {
				acc /= a[l * acs + l];
			}
			x[l] = acc;
		}
	} else {
		for l in 0..m {
			let mut acc = x[l];
			for i in 0..l {
				acc += a[l * acs + i] * -x[i];
			}
			if !unit {
				acc /= a[l * acs + l];
			}
			x[l] = acc;
		}
	}
}
