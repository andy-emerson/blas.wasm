//! `strsm` — triangular solve, many right-hand sides, in place:
//! B ← αA⁻¹B (left) or B ← αBA⁻¹ (right).
//!
//! Implementation: both sides on the packing layer (`strsm_left_packed`,
//! `strsm_right_packed`: the solved rows or columns enter through the 8×4
//! broadcast tile from panels appended as each block is solved; the
//! diagonal block is solved on lanes on the left, by the sweep's in-group
//! elimination and reciprocal on the right); the sweeps stay exported as
//! `strsm_left_sweep` and `strsm_right_sweep`. The transposed forms route
//! the no-copy kernels (`strsm_stream`) over the workspace copy
//! (`strsm_copy`), per the race.
//!
//! The family's chapter — what the operation is, exactly which rounding
//! sequence every shape reproduces, how it is computed, what it calls and
//! how it performs — is this folder's README, the `trsm` module page.

use super::super::check_mat;
use super::super::dtrmm::right_panel_base;
use super::super::sgemm::{tile_8x4_packed_bcast, KC, MC, MR};
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
/// to per side is the race's verdict (`bench/README.md`; decision D3 in `DESIGN.md`).
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
		(Side::Left, Trans::NoTrans) => strsm_left_packed(uplo.is_upper(), diag.is_unit(), alpha, m, n, a, lda, b, ldb, work),
		(Side::Right, Trans::NoTrans) => strsm_right_packed(uplo.is_upper(), diag.is_unit(), alpha, m, n, a, lda, b, ldb, work),
		(Side::Left, _) if LEFT_T_STREAM => strsm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		(Side::Right, _) if RIGHT_T_STREAM => strsm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		_ => strsm_copy(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, work),
	}
	Ok(())
}

/// Elements of scratch [`strsm`] needs for these flags and dimensions:
/// the packed panels for the no-transpose forms ([`strsm_left_packed_work`]
/// / [`strsm_right_packed_work`]); zero for a transposed form routed to
/// the streaming candidate; the `na×na` copy plus the panels
/// ([`strsm_copy_work`]) where the copy candidate is routed.
pub fn strsm_work(side: Side, transa: Trans, m: usize, n: usize) -> usize {
	if m == 0 || n == 0 {
		return 0;
	}
	if transa == Trans::NoTrans {
		return if side == Side::Left { strsm_left_packed_work(m, n) } else { strsm_right_packed_work(m, n) };
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

/// Candidate for the transposed forms (decision D3 in `DESIGN.md`; the race in `bench/README.md`): op(A)
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
	assert!(work.len() >= strsm_copy_work(side, m, n), "strsm_copy: workspace shorter than strsm_copy_work");
	let (ta, rest) = work.split_at_mut(na * na);
		super::super::transpose_into(a, lda, na, na, ta, |x| x);
	let upper = !uplo.is_upper();
	match side {
		Side::Left => strsm_left_packed(upper, diag.is_unit(), alpha, m, n, ta, na.max(1), b, ldb, rest),
		Side::Right => strsm_right_packed(upper, diag.is_unit(), alpha, m, n, ta, na.max(1), b, ldb, rest),
	}
}

/// Elements of scratch [`strsm_copy`] needs: the `na×na` copy of
/// op(A), `na` being `m` on the left and `n` on the right, plus the
/// packed kernel's panels.
pub fn strsm_copy_work(side: Side, m: usize, n: usize) -> usize {
	let na = if side == Side::Left { m } else { n };
	na * na + if side == Side::Left { strsm_left_packed_work(m, n) } else { strsm_right_packed_work(m, n) }
}

/// Candidate for the transposed forms (decision D3 in `DESIGN.md`; the race in `bench/README.md`): no
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

/// The K1 candidate for the left side: `B ← α·A⁻¹·B` on the packing
/// layer, in place. B is scaled by α first, as the sweep does. The
/// k-blocks run in the elimination's direction (ascending for lower,
/// descending for upper); inside a k-block the row blocks are solved in
/// that direction too: a tile first takes the block's rows past the
/// already-solved rows of this k-block (their negated solutions sit in
/// the right panel, appended as each block is solved), then a scalar
/// routine solves the MR×MR diagonal block in the sweep's order, then
/// the new rows are negated into the panel; once the k-block is solved
/// the remaining row blocks take the whole panel with the tile. The
/// `m % MR` tail rows are solved by the plain sweep (after the main loop
/// for lower, before it for upper, where they feed the first panel);
/// the `n % 4` column tail runs `dtrsv` per column. Per-element rounding
/// sequence identical to the shipping sweep (α·b, then the subtractions
/// in elimination order as `(−x_l)·a_il`, then the division):
/// bit-for-bit interchangeable, tested. Raced against the shipping
/// sweep and against OpenBLAS compiled to wasm (`bench/README.md`).
#[allow(clippy::too_many_arguments)]
pub fn strsm_left_packed(upper: bool, unit: bool, alpha: f32, m: usize, n: usize, a: &[f32], acs: usize, b: &mut [f32], bcs: usize, work: &mut [f32]) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let need = strsm_left_packed_work(m, n);
	assert!(work.len() >= need, "strsm_left_packed: workspace shorter than strsm_left_packed_work");
	if m == 0 || n == 0 {
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	for j in n4..n {
		let col = &mut b[j * bcs..j * bcs + m];
		if alpha != 1.0 {
			sscal(alpha, col);
		}
		strsv_k(m, a, acs, upper, unit, col);
	}
	if n4 == 0 {
		return;
	}
	if alpha != 1.0 {
		for j in 0..n4 {
			sscal(alpha, &mut b[j * bcs..j * bcs + m]);
		}
	}
	let at = |i: usize, l: usize| a[l * acs + i];
	// the tail rows by the plain sweep: lower after the main loop (they
	// read every solved row above), upper before it (they are the first
	// unknowns and feed the first panel)
	let row_tail = |b: &mut [f32]| {
		for j in 0..n4 {
			for step in 0..m - m_mr {
				let r = if upper { m - 1 - step } else { m_mr + step };
				let bj = &b[j * bcs..j * bcs + m];
				let mut acc = bj[r];
				if upper {
					for (l, &v) in bj.iter().enumerate().skip(r + 1).rev() {
						acc += -v * at(r, l);
					}
				} else {
					for (l, &v) in bj.iter().enumerate().take(r) {
						acc += -v * at(r, l);
					}
				}
				b[j * bcs + r] = if unit { acc } else { acc / at(r, r) };
			}
		}
	};
	if upper {
		row_tail(b);
	}
	if m_mr > 0 {
		let kcm = KC.min(m);
		let (bp, ap) = work[..need].split_at_mut(kcm * n4);
		let nblk = m.div_ceil(KC);
		// a row's slot in the (reversed for upper) panel of its k-block
		let slot = |l: usize, pc: usize, kc: usize| -> usize { if upper { pc + kc - 1 - l } else { l - pc } };
		// negate a solved row into the panel of every column group
		let append = |bp: &mut [f32], b: &[f32], l: usize, pc: usize, kc: usize| {
			let t = slot(l, pc, kc);
			for jg in 0..n4 / 4 {
				for u in 0..4 {
					bp[jg * kc * 4 + t * 4 + u] = -b[(jg * 4 + u) * bcs + l];
				}
			}
		};
		for bi in 0..nblk {
			let blk = if upper { nblk - 1 - bi } else { bi };
			let pc = blk * KC;
			let kc = KC.min(m - pc);
			// the tail rows, already solved, belong to the top k-block's panel
			if upper {
				for l in m_mr.max(pc)..pc + kc {
					append(bp, b, l, pc, kc);
				}
			}
			// the row blocks of this k-block, in elimination order
			let blocks_here = (pc..(pc + kc).min(m_mr)).step_by(MR).count();
			for bj in 0..blocks_here {
				let r0 = if upper { pc + (blocks_here - 1 - bj) * MR } else { pc + bj * MR };
				// (a) the solved rows of this k-block so far
				let solved = if upper { pc + kc - (r0 + MR) } else { r0 - pc };
				if solved > 0 {
					let base = 0;
					for t in 0..solved {
						let l = if upper { pc + kc - 1 - t } else { pc + t };
						let src = l * acs + r0;
						ap[base + t * MR..base + t * MR + MR].copy_from_slice(&a[src..src + MR]);
					}
					for jg in 0..n4 / 4 {
						unsafe {
							tile_8x4_packed_bcast(solved, ap.as_ptr(), bp.as_ptr().add(jg * kc * 4), 1.0, false, b.as_mut_ptr().add(jg * 4 * bcs + r0), bcs);
						}
					}
				}
				// (b) the diagonal block, in the sweep's order, four columns per
				// lane group
				for jg in 0..n4 / 4 {
					unsafe {
						diag_solve_rows(upper, unit, r0, a, acs, b.as_mut_ptr().add(jg * 4 * bcs), bcs);
					}
				}
				// (c) into the panel
				for l in r0..r0 + MR {
					append(bp, b, l, pc, kc);
				}
			}
			// the row blocks past this k-block take the whole panel
			let mut ic = 0usize;
			while ic < m_mr {
				let mc = MC.min(m_mr - ic);
				for g in 0..mc / MR {
					let r0 = ic + g * MR;
					let past = if upper { r0 + MR <= pc } else { r0 >= pc + kc };
					if !past {
						continue;
					}
					let base = g * kc * MR;
					for t in 0..kc {
						let l = if upper { pc + kc - 1 - t } else { pc + t };
						let src = l * acs + r0;
						ap[base + t * MR..base + t * MR + MR].copy_from_slice(&a[src..src + MR]);
					}
					for jg in 0..n4 / 4 {
						unsafe {
							tile_8x4_packed_bcast(kc, ap.as_ptr().add(base), bp.as_ptr().add(jg * kc * 4), 1.0, false, b.as_mut_ptr().add(jg * 4 * bcs + r0), bcs);
						}
					}
				}
				ic += mc;
			}
		}
	}
	if !upper {
		row_tail(b);
	}
}

/// Elements of scratch [`strsm_left_packed`] needs: the right panel and
/// the left panel; zero when the packed loops have nothing to do.
pub fn strsm_left_packed_work(m: usize, n: usize) -> usize {
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	if m_mr == 0 || n4 == 0 {
		0
	} else {
		let kc = KC.min(m);
		kc * n4 + MC.min(m_mr) * kc
	}
}

/// The K1 candidate for the right side: `B ← α·B·A⁻¹` on the packing
/// layer, in place. Groups of four columns are solved in the sweep's
/// order; each group is scaled by α, takes the already-solved columns'
/// terms through the tile in ascending k (their solutions sit in the
/// left panels, appended as each group is solved; the `m % MR` tail
/// rows take the same terms by column-axpy from the solved columns),
/// then finishes with the sweep's own in-group elimination and the
/// reciprocal scaling; the tail columns run the sweep last. Per-element
/// rounding sequence identical to the shipping sweep: bit-for-bit
/// interchangeable, tested. Raced against the shipping sweep and
/// against OpenBLAS compiled to wasm (`bench/README.md`).
#[allow(clippy::too_many_arguments)]
pub fn strsm_right_packed(upper: bool, unit: bool, alpha: f32, m: usize, n: usize, a: &[f32], acs: usize, b: &mut [f32], bcs: usize, work: &mut [f32]) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	let need = strsm_right_packed_work(m, n);
	assert!(work.len() >= need, "strsm_right_packed: workspace shorter than strsm_right_packed_work");
	if m == 0 || n == 0 {
		return;
	}
	let at = |row: usize, col: usize| a[col * acs + row];
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	let mt = m - m_mr;
	// the sweep's grouping: upper ascends over start-aligned groups with
	// the tail at the end; lower descends over end-aligned groups with
	// the tail columns 0..off done last
	let off = if upper { 0 } else { n % 4 };
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref = |idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const f32, m) };
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
	if n4 > 0 {
		let (lp, rp) = work[..need].split_at_mut(m_mr * n);
		let nblk = n.div_ceil(KC);
		let append = |lp: &mut [f32], b: &[f32], k: usize| {
			let blk = k / KC;
			let pc = blk * KC;
			let kc = KC.min(n - pc);
			let base = right_panel_base(blk, n, m_mr);
			for g in 0..m_mr / MR {
				let src = k * bcs + g * MR;
				lp[base + g * kc * MR + (k - pc) * MR..base + g * kc * MR + (k - pc) * MR + MR].copy_from_slice(&b[src..src + MR]);
			}
		};
		for q in 0..n4 / 4 {
			let gs = if upper { q * 4 } else { off + (n4 / 4 - 1 - q) * 4 };
			if alpha != 1.0 {
				for u in 0..4 {
					sscal(alpha, col_mut(gs + u));
				}
			}
			// the solved columns' terms, ascending k
			let (ka, kb) = if upper { (0, gs) } else { (gs + 4, n) };
			for blk in 0..nblk {
				let pc = blk * KC;
				let kc = KC.min(n - pc);
				let (k0, k1) = (ka.max(pc), kb.min(pc + kc));
				if k0 >= k1 {
					continue;
				}
				let dst = &mut rp[..(k1 - k0) * 4];
				for (t, k) in (k0..k1).enumerate() {
					for u in 0..4 {
						dst[t * 4 + u] = -at(k, gs + u);
					}
				}
				let base = right_panel_base(blk, n, m_mr);
				for g in 0..m_mr / MR {
					unsafe {
						tile_8x4_packed_bcast(
							k1 - k0,
							lp.as_ptr().add(base + g * kc * MR + (k0 - pc) * MR),
							dst.as_ptr(),
							1.0,
							false,
							b.as_mut_ptr().add(gs * bcs + g * MR),
							bcs,
						);
					}
				}
				if mt > 0 {
					for (t, k) in (k0..k1).enumerate() {
						for u in 0..4 {
							let (src, dstc) = unsafe { (core::slice::from_raw_parts(bp.add(k * bcs + m_mr) as *const f32, mt), core::slice::from_raw_parts_mut(bp.add((gs + u) * bcs + m_mr), mt)) };
							saxpy_k(dst[t * 4 + u], src, dstc);
						}
					}
				}
			}
			// in-group elimination, the sweep's order, then the reciprocal
			if upper {
				for tc in gs..gs + 4 {
					for k in gs..tc {
						saxpy_k(-at(k, tc), col_ref(k), col_mut(tc));
					}
					if !unit {
						sscal(1.0 / at(tc, tc), col_mut(tc));
					}
				}
			} else {
				for tc in (gs..gs + 4).rev() {
					for k in tc + 1..gs + 4 {
						saxpy_k(-at(k, tc), col_ref(k), col_mut(tc));
					}
					if !unit {
						sscal(1.0 / at(tc, tc), col_mut(tc));
					}
				}
			}
			for k in gs..gs + 4 {
				append(lp, b, k);
			}
		}
	}
	// tail columns last: they read every solved column
	if upper {
		for j in n4..n {
			do_col(j, 0, j);
		}
	} else {
		for j in (0..off).rev() {
			do_col(j, j + 1, n);
		}
	}
}

/// Elements of scratch [`strsm_right_packed`] needs: the solved
/// columns' packed rows and one right panel.
pub fn strsm_right_packed_work(m: usize, n: usize) -> usize {
	let n4 = n - n % 4;
	if m == 0 || n4 == 0 {
		0
	} else {
		(m - m % MR) * n + KC.min(n) * 4
	}
}

/// The diagonal MR×MR solve of the packed left trsm for one 4-column
/// group (the f32 twin: the four columns ride one lane quad).
/// # Safety
/// As the f64 twin in `dtrsm`.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn diag_solve_rows(upper: bool, unit: bool, r0: usize, a: &[f32], acs: usize, b: *mut f32, bcs: usize) {
	use crate::lanes::F32x4;
	let mut x = [F32x4::splat(0.0); MR];
	for s in 0..MR {
		let i = if upper { r0 + MR - 1 - s } else { r0 + s };
		let mut v = F32x4::quad(*b.add(i), *b.add(bcs + i), *b.add(2 * bcs + i), *b.add(3 * bcs + i));
		for q in 0..s {
			let l = if upper { r0 + MR - 1 - q } else { r0 + q };
			v = v.add(x[l - r0].neg().mul(F32x4::splat(a[l * acs + i])));
		}
		if !unit {
			v = v.div(F32x4::splat(a[i * acs + i]));
		}
		x[i - r0] = v;
		*b.add(i) = v.lane0();
		*b.add(bcs + i) = v.lane1();
		*b.add(2 * bcs + i) = v.lane2();
		*b.add(3 * bcs + i) = v.lane3();
	}
}

/// The shipping left shape before K1, the lockstep column sweep: kept
/// as the race's loser and the bit-identity reference of
/// [`strsm_left_packed`]; exported so the tests and the benchmarks can
/// call it. Asserts rather than returning errors.
#[allow(clippy::too_many_arguments)]
pub fn strsm_left_sweep(upper: bool, unit: bool, alpha: f32, m: usize, n: usize, a: &[f32], lda: usize, b: &mut [f32], ldb: usize) {
	strsm_left_k(alpha, m, n, a, lda, upper, unit, b, ldb);
}

/// The shipping right shape before K1, the grouped column sweep: kept
/// as the race's loser and the bit-identity reference of
/// [`strsm_right_packed`]; exported so the tests and the benchmarks can
/// call it. Asserts rather than returning errors.
#[allow(clippy::too_many_arguments)]
pub fn strsm_right_sweep(upper: bool, unit: bool, alpha: f32, m: usize, n: usize, a: &[f32], lda: usize, b: &mut [f32], ldb: usize) {
	strsm_right_k(alpha, m, n, a, lda, upper, unit, b, ldb, Trans::NoTrans);
}
