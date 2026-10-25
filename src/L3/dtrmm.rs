//! `dtrmm` — triangular matrix multiply, in place:
//! B ← αAB (left) or B ← αBA (right), A triangular.
//!
//! Implementation: left side walks four B columns
//! in lockstep through the `dtrmv` column sweep, so each A column
//! streams once per group of four instead of once per column (A
//! traffic 4× down; per-column rounding sequence unchanged — the
//! columns are independent — so results are bit-for-bit the plain
//! dtrmv-per-column loop, tested). Right side groups four destination
//! columns: contributions from inside the group run first in the
//! original elimination order (descending for upper, ascending for
//! lower — each original column consumed before it is overwritten),
//! then the columns outside the group fan out via one stream each.
//! For LOWER the resulting per-element add order is identical to the
//! plain sweep (ascending k throughout); for UPPER the in-group adds
//! now precede the out-of-group adds — a deterministic, documented
//! reorder (the tests' scalar replay mirrors it). A fused-FMA variant
//! measured better in the campaign record step-1 race but is DEFERRED at the f64
//! campaign close: wasm relaxed-madd rounding is implementation-
//! dependent, so it trades away cross-target bit-identity — an
//! architect decision; DESIGN.md, "Fast mode is a lever". The transposed forms route
//! between a workspace copy and the no-copy kernels per the race
//! (`bench/README.md`).

use super::check_mat;
use super::dgemm::{tile_4x4_packed_bcast, KC, MC, MR};
use crate::{Diag, Error, Side, Trans, Uplo};
use crate::kernels::daxpy4;
use crate::L1::{daxpy_k, dscal};
use crate::L2::dtrmv_k;

/// `B ← α·op(A)·B` ([`Side::Left`]) or `B ← α·B·op(A)` ([`Side::Right`])
/// for a triangular `A`, the triangle named by `uplo` stored, with a
/// unit diagonal when `diag` says so; `A` is `m×m` on the left and `n×n`
/// on the right, `B` is `m×n`, each at its own leading dimension.
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ` per `transa`. `work` is scratch of at
/// least [`dtrmm_work`]`(side, transa, m, n)` elements.
///
/// The transposed forms are defined as the no-transpose form on an
/// explicit transpose; `ConjTrans` is `Trans` for a real matrix, and two
/// candidates compute them, [`dtrmm_copy`] and [`dtrmm_stream`],
/// both bit-identical to that definition; which one this routine routes
/// to per side is the race's verdict (`bench/README.md`; decision D3 in `DESIGN.md`).
///
/// Errors: a leading dimension below its row count (arguments 9, 11);
/// `a` or `b` shorter than its extent (arguments 8, 10); `work` shorter
/// than the query (argument 12).
#[allow(clippy::too_many_arguments)]
pub fn dtrmm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize, b: &mut [f64], ldb: usize, work: &mut [f64]) -> Result<(), Error> {
	const R: &str = "dtrmm";
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 8, 9, a.len(), na, na, lda)?;
	crate::error::mat(R, 10, 11, b.len(), m, n, ldb)?;
	crate::error::work(R, 12, work.len(), dtrmm_work(side, transa, m, n))?;
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
		(Side::Left, Trans::NoTrans) => dtrmm_left_packed(uplo.is_upper(), diag.is_unit(), alpha, m, n, a, lda, b, ldb, work),
		(Side::Right, Trans::NoTrans) => dtrmm_right_packed(uplo.is_upper(), diag.is_unit(), alpha, m, n, a, lda, b, ldb, work),
		(Side::Left, _) if LEFT_T_STREAM => dtrmm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		(Side::Right, _) if RIGHT_T_STREAM => dtrmm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		_ => dtrmm_copy(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, work),
	}
	Ok(())
}

/// Elements of scratch [`dtrmm`] needs for these flags and dimensions:
/// the packed panels for the no-transpose forms ([`dtrmm_left_packed_work`]
/// / [`dtrmm_right_packed_work`]); zero for a transposed form routed to
/// the streaming candidate; the `na×na` copy plus the panels
/// ([`dtrmm_copy_work`]) where the copy candidate is routed.
pub fn dtrmm_work(side: Side, transa: Trans, m: usize, n: usize) -> usize {
	if m == 0 || n == 0 {
		return 0;
	}
	if transa == Trans::NoTrans {
		return if side == Side::Left { dtrmm_left_packed_work(m, n) } else { dtrmm_right_packed_work(m, n) };
	}
	let stream = if side == Side::Left { LEFT_T_STREAM } else { RIGHT_T_STREAM };
	if stream { 0 } else { dtrmm_copy_work(side, m, n) }
}

// Which candidate the transposed forms route to, per side — the race's
// verdict (two runner draws, `bench/README.md`): the stream won both
// sides in the real types (left 1.3–2.8× for f64 at every size from 256,
// f32 1.0–2.4× with the ascending-walk cases at 0.85–1.0 on several
// square shapes; right 1.02–1.6×), needing no workspace. Both candidates
// stay exported and tested.
const LEFT_T_STREAM: bool = true;
const RIGHT_T_STREAM: bool = true;

/// B ← αAB. A is m×m triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dtrmm_left_k(
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
		let bp = b.as_mut_ptr();
		let cols: [*mut f64; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// the dtrmv column sweep, four x vectors in lockstep sharing
		// each A column read
		if upper {
			for l in 0..m {
				let t = unsafe { [*cols[0].add(l), *cols[1].add(l), *cols[2].add(l), *cols[3].add(l)] };
				unsafe {
					daxpy4(a.as_ptr().add(l * acs), t, cols[0], cols[1], cols[2], cols[3], l);
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
				let t = unsafe { [*cols[0].add(l), *cols[1].add(l), *cols[2].add(l), *cols[3].add(l)] };
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
				if !unit {
					let d = a[l * acs + l];
					for (u, cu) in cols.iter().enumerate() {
						unsafe { *cu.add(l) = t[u] * d };
					}
				}
			}
		}
		if alpha != 1.0 {
			for u in 0..4 {
				dscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		j += 4;
	}
	while j < n {
		let col = &mut b[j * bcs..j * bcs + m];
		dtrmv_k(m, a, acs, upper, unit, col);
		if alpha != 1.0 {
			dscal(alpha, col);
		}
		j += 1;
	}
}

/// B ← αBA. A is n×n triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dtrmm_right_k(
	alpha: f64,
	m: usize,
	n: usize,
	a: &[f64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [f64],
	bcs: usize,
	op: Trans,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	// `upper` and the accessor describe op(A): for a transposed op the
	// stored triangle is the other one and the element roles swap
	let at = |row: usize, col: usize| -> f64 {
		if op == Trans::NoTrans {
			a[col * acs + row]
		} else {
			a[row * acs + col]
		}
	};
	// result column j = α·Σ_{k in triangle} a`[k,j]`·B_orig[:,k]. The
	// column order (descending for upper, ascending for lower) touches
	// each original column before anything overwrites it; groups of
	// four keep that order internally, then share one stream per
	// out-of-group source column. Columns j and k never alias
	// (k ≠ j always), so the views below are disjoint — raw pointers
	// express what split_at_mut can't say cleanly for arbitrary
	// (j, k) pairs.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const f64, m) };
	let do_col = |j: usize| {
		let diag = if unit { 1.0 } else { at(j, j) };
		dscal(alpha * diag, col_mut(j));
		let (lo, hi) = if upper { (0, j) } else { (j + 1, n) };
		for k in lo..hi {
			daxpy_k(alpha * at(k, j), col_ref(k), col_mut(j));
		}
	};
	let fan_out = |gs: usize, k: usize| {
		let t = [
			alpha * at(k, gs),
			alpha * at(k, gs + 1),
			alpha * at(k, gs + 2),
			alpha * at(k, gs + 3),
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
		let r = n % 4;
		let mut gs = n;
		while gs >= r + 4 {
			gs -= 4;
			// in-group first, original descending order (sources still
			// original when read)
			for tc in (gs..gs + 4).rev() {
				let diag = if unit { 1.0 } else { at(tc, tc) };
				dscal(alpha * diag, col_mut(tc));
				for k in gs..tc {
					daxpy_k(alpha * at(k, tc), col_ref(k), col_mut(tc));
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
				let diag = if unit { 1.0 } else { at(tc, tc) };
				dscal(alpha * diag, col_mut(tc));
				for k in tc + 1..gs + 4 {
					daxpy_k(alpha * at(k, tc), col_ref(k), col_mut(tc));
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

/// Candidate for the transposed forms (decision D3 in `DESIGN.md`; the race in `bench/README.md`): op(A)
/// is transposed into `work` and the no-transpose
/// kernel runs on the copy with the triangle flipped, so the result is
/// the explicit-copy definition itself. `work` needs
/// [`dtrmm_copy_work`]`(side, m, n)` elements. Exported so the
/// tests and the benchmarks can race it; it asserts rather than
/// returning errors, and `transa` must not be `NoTrans`.
#[allow(clippy::too_many_arguments)]
pub fn dtrmm_copy(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize, b: &mut [f64], ldb: usize, work: &mut [f64]) {
	assert!(transa != Trans::NoTrans, "dtrmm_copy: a transposed form only");
	let na = if side == Side::Left { m } else { n };
	check_mat(a.len(), na, na, lda);
	assert!(work.len() >= dtrmm_copy_work(side, m, n), "dtrmm_copy: workspace shorter than dtrmm_copy_work");
	let (ta, rest) = work.split_at_mut(na * na);
	super::transpose_into(a, lda, na, na, ta, |x| x);
	let upper = !uplo.is_upper();
	match side {
		Side::Left => dtrmm_left_packed(upper, diag.is_unit(), alpha, m, n, ta, na.max(1), b, ldb, rest),
		Side::Right => dtrmm_right_packed(upper, diag.is_unit(), alpha, m, n, ta, na.max(1), b, ldb, rest),
	}
}

/// Elements of scratch [`dtrmm_copy`] needs: the `na×na` copy of
/// op(A), `na` being `m` on the left and `n` on the right, plus the
/// packed kernel's panels.
pub fn dtrmm_copy_work(side: Side, m: usize, n: usize) -> usize {
	let na = if side == Side::Left { m } else { n };
	na * na + if side == Side::Left { dtrmm_left_packed_work(m, n) } else { dtrmm_right_packed_work(m, n) }
}

/// Candidate for the transposed forms (decision D3 in `DESIGN.md`; the race in `bench/README.md`): no
/// copy; on the right the no-transpose kernel reads A through a
/// transposed accessor (it only ever reads A as scalars), on the left
/// `dtrmm_left_t_k` reduces down the stored columns. Bit-identical
/// to [`dtrmm_copy`] (tested). Exported so the tests and the
/// benchmarks can race it; it asserts rather than returning errors, and
/// `transa` must not be `NoTrans`.
#[allow(clippy::too_many_arguments)]
pub fn dtrmm_stream(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: f64, a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {
	assert!(transa != Trans::NoTrans, "dtrmm_stream: a transposed form only");
	let upper = !uplo.is_upper();
	match side {
		Side::Left => dtrmm_left_t_k(alpha, m, n, a, lda, upper, diag.is_unit(), b, ldb),
		Side::Right => dtrmm_right_k(alpha, m, n, a, lda, upper, diag.is_unit(), b, ldb, transa),
	}
}

/// B ← α·op(A)·B with op(A) = Aᵀ, computed from the stored A
/// without a copy: `upper` names op(A)'s triangle (stored A holds the
/// other). Each element's arithmetic sequence is the lockstep
/// dtrmv sweep's — the diagonal product first, then the products added one at a
/// time down the stored column (ascending for upper, descending for
/// lower), then the α scaling — so the result is bit-identical to
/// [`dtrmm_left_k`] on an explicit transpose (tested). The four
/// right-hand sides of a group ride the lanes, and rows go four at a
/// time so one step loads four A values and four B values for sixteen
/// multiply-adds; op(A)'s row l is stored column l, read contiguously.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dtrmm_left_t_k(
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
		let bp = b.as_mut_ptr();
		let c: [*mut f64; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		unsafe { dtrmm_left_t4(m, a, acs, upper, unit, c) };
		if alpha != 1.0 {
			for u in 0..4 {
				dscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		j += 4;
	}
	while j < n {
		let col = &mut b[j * bcs..j * bcs + m];
		dtrmm_left_t1(m, a, acs, upper, unit, col);
		if alpha != 1.0 {
			dscal(alpha, col);
		}
		j += 1;
	}
}

/// Four right-hand sides of [`dtrmm_left_t_k`] in lockstep.
/// # Safety
/// Each column pointer must be valid for `m` elements; the columns must
/// not alias each other.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn dtrmm_left_t4(m: usize, a: &[f64], acs: usize, upper: bool, unit: bool, c: [*mut f64; 4]) {
	use crate::lanes::F64x2;
	let ap = a.as_ptr();
	if upper {
		let mut lb = 0usize;
		while lb + 4 <= m {
			let mut acc: [[F64x2; 2]; 4] = [[F64x2::pair(*c[0].add(lb), *c[1].add(lb)), F64x2::pair(*c[2].add(lb), *c[3].add(lb))], [F64x2::pair(*c[0].add(lb + 1), *c[1].add(lb + 1)), F64x2::pair(*c[2].add(lb + 1), *c[3].add(lb + 1))], [F64x2::pair(*c[0].add(lb + 2), *c[1].add(lb + 2)), F64x2::pair(*c[2].add(lb + 2), *c[3].add(lb + 2))], [F64x2::pair(*c[0].add(lb + 3), *c[1].add(lb + 3)), F64x2::pair(*c[2].add(lb + 3), *c[3].add(lb + 3))]];
			let x: [[F64x2; 2]; 4] = acc;
			if !unit {
				let d = F64x2::splat(*ap.add((lb) * acs + lb));
				acc[0][0] = acc[0][0].mul(d);
				acc[0][1] = acc[0][1].mul(d);
			}
			if !unit {
				let d = F64x2::splat(*ap.add((lb + 1) * acs + lb + 1));
				acc[1][0] = acc[1][0].mul(d);
				acc[1][1] = acc[1][1].mul(d);
			}
			if !unit {
				let d = F64x2::splat(*ap.add((lb + 2) * acs + lb + 2));
				acc[2][0] = acc[2][0].mul(d);
				acc[2][1] = acc[2][1].mul(d);
			}
			if !unit {
				let d = F64x2::splat(*ap.add((lb + 3) * acs + lb + 3));
				acc[3][0] = acc[3][0].mul(d);
				acc[3][1] = acc[3][1].mul(d);
			}
			let av = F64x2::splat(*ap.add((lb) * acs + lb + 1));
			acc[0][0] = acc[0][0].add(av.mul(x[1][0]));
			acc[0][1] = acc[0][1].add(av.mul(x[1][1]));
			let av = F64x2::splat(*ap.add((lb) * acs + lb + 2));
			acc[0][0] = acc[0][0].add(av.mul(x[2][0]));
			acc[0][1] = acc[0][1].add(av.mul(x[2][1]));
			let av = F64x2::splat(*ap.add((lb + 1) * acs + lb + 2));
			acc[1][0] = acc[1][0].add(av.mul(x[2][0]));
			acc[1][1] = acc[1][1].add(av.mul(x[2][1]));
			let av = F64x2::splat(*ap.add((lb) * acs + lb + 3));
			acc[0][0] = acc[0][0].add(av.mul(x[3][0]));
			acc[0][1] = acc[0][1].add(av.mul(x[3][1]));
			let av = F64x2::splat(*ap.add((lb + 1) * acs + lb + 3));
			acc[1][0] = acc[1][0].add(av.mul(x[3][0]));
			acc[1][1] = acc[1][1].add(av.mul(x[3][1]));
			let av = F64x2::splat(*ap.add((lb + 2) * acs + lb + 3));
			acc[2][0] = acc[2][0].add(av.mul(x[3][0]));
			acc[2][1] = acc[2][1].add(av.mul(x[3][1]));
			for i in lb + 4..m {
				let t0 = F64x2::pair(*c[0].add(i), *c[1].add(i));
				let t1 = F64x2::pair(*c[2].add(i), *c[3].add(i));
				for (v, row) in acc.iter_mut().enumerate() {
					let av = F64x2::splat(*ap.add((lb + v) * acs + i));
					row[0] = row[0].add(av.mul(t0));
					row[1] = row[1].add(av.mul(t1));
				}
			}
			*c[0].add(lb) = acc[0][0].lane0();
			*c[1].add(lb) = acc[0][0].lane1();
			*c[2].add(lb) = acc[0][1].lane0();
			*c[3].add(lb) = acc[0][1].lane1();
			*c[0].add(lb + 1) = acc[1][0].lane0();
			*c[1].add(lb + 1) = acc[1][0].lane1();
			*c[2].add(lb + 1) = acc[1][1].lane0();
			*c[3].add(lb + 1) = acc[1][1].lane1();
			*c[0].add(lb + 2) = acc[2][0].lane0();
			*c[1].add(lb + 2) = acc[2][0].lane1();
			*c[2].add(lb + 2) = acc[2][1].lane0();
			*c[3].add(lb + 2) = acc[2][1].lane1();
			*c[0].add(lb + 3) = acc[3][0].lane0();
			*c[1].add(lb + 3) = acc[3][0].lane1();
			*c[2].add(lb + 3) = acc[3][1].lane0();
			*c[3].add(lb + 3) = acc[3][1].lane1();
			lb += 4;
		}
		for l in lb..m {
			let mut acc: [F64x2; 2] = [F64x2::pair(*c[0].add(l), *c[1].add(l)), F64x2::pair(*c[2].add(l), *c[3].add(l))];
			if !unit {
				let d = F64x2::splat(*ap.add(l * acs + l));
				acc[0] = acc[0].mul(d);
				acc[1] = acc[1].mul(d);
			}
			for i in l + 1..m {
				let t0 = F64x2::pair(*c[0].add(i), *c[1].add(i));
				let t1 = F64x2::pair(*c[2].add(i), *c[3].add(i));
				let av = F64x2::splat(*ap.add(l * acs + i));
				acc[0] = acc[0].add(av.mul(t0));
				acc[1] = acc[1].add(av.mul(t1));
			}
			*c[0].add(l) = acc[0].lane0();
			*c[1].add(l) = acc[0].lane1();
			*c[2].add(l) = acc[1].lane0();
			*c[3].add(l) = acc[1].lane1();
		}
	} else {
		let r = m % 4;
		let mut lb = m;
		while lb >= r + 4 {
			lb -= 4;
			let mut acc: [[F64x2; 2]; 4] = [[F64x2::pair(*c[0].add(lb), *c[1].add(lb)), F64x2::pair(*c[2].add(lb), *c[3].add(lb))], [F64x2::pair(*c[0].add(lb + 1), *c[1].add(lb + 1)), F64x2::pair(*c[2].add(lb + 1), *c[3].add(lb + 1))], [F64x2::pair(*c[0].add(lb + 2), *c[1].add(lb + 2)), F64x2::pair(*c[2].add(lb + 2), *c[3].add(lb + 2))], [F64x2::pair(*c[0].add(lb + 3), *c[1].add(lb + 3)), F64x2::pair(*c[2].add(lb + 3), *c[3].add(lb + 3))]];
			let x: [[F64x2; 2]; 4] = acc;
			if !unit {
				let d = F64x2::splat(*ap.add((lb) * acs + lb));
				acc[0][0] = acc[0][0].mul(d);
				acc[0][1] = acc[0][1].mul(d);
			}
			if !unit {
				let d = F64x2::splat(*ap.add((lb + 1) * acs + lb + 1));
				acc[1][0] = acc[1][0].mul(d);
				acc[1][1] = acc[1][1].mul(d);
			}
			if !unit {
				let d = F64x2::splat(*ap.add((lb + 2) * acs + lb + 2));
				acc[2][0] = acc[2][0].mul(d);
				acc[2][1] = acc[2][1].mul(d);
			}
			if !unit {
				let d = F64x2::splat(*ap.add((lb + 3) * acs + lb + 3));
				acc[3][0] = acc[3][0].mul(d);
				acc[3][1] = acc[3][1].mul(d);
			}
			let av = F64x2::splat(*ap.add((lb + 3) * acs + lb + 2));
			acc[3][0] = acc[3][0].add(av.mul(x[2][0]));
			acc[3][1] = acc[3][1].add(av.mul(x[2][1]));
			let av = F64x2::splat(*ap.add((lb + 2) * acs + lb + 1));
			acc[2][0] = acc[2][0].add(av.mul(x[1][0]));
			acc[2][1] = acc[2][1].add(av.mul(x[1][1]));
			let av = F64x2::splat(*ap.add((lb + 3) * acs + lb + 1));
			acc[3][0] = acc[3][0].add(av.mul(x[1][0]));
			acc[3][1] = acc[3][1].add(av.mul(x[1][1]));
			let av = F64x2::splat(*ap.add((lb + 1) * acs + lb));
			acc[1][0] = acc[1][0].add(av.mul(x[0][0]));
			acc[1][1] = acc[1][1].add(av.mul(x[0][1]));
			let av = F64x2::splat(*ap.add((lb + 2) * acs + lb));
			acc[2][0] = acc[2][0].add(av.mul(x[0][0]));
			acc[2][1] = acc[2][1].add(av.mul(x[0][1]));
			let av = F64x2::splat(*ap.add((lb + 3) * acs + lb));
			acc[3][0] = acc[3][0].add(av.mul(x[0][0]));
			acc[3][1] = acc[3][1].add(av.mul(x[0][1]));
			let mut i = lb;
			while i > 0 {
				i -= 1;
				let t0 = F64x2::pair(*c[0].add(i), *c[1].add(i));
				let t1 = F64x2::pair(*c[2].add(i), *c[3].add(i));
				for (v, row) in acc.iter_mut().enumerate() {
					let av = F64x2::splat(*ap.add((lb + v) * acs + i));
					row[0] = row[0].add(av.mul(t0));
					row[1] = row[1].add(av.mul(t1));
				}
			}
			*c[0].add(lb) = acc[0][0].lane0();
			*c[1].add(lb) = acc[0][0].lane1();
			*c[2].add(lb) = acc[0][1].lane0();
			*c[3].add(lb) = acc[0][1].lane1();
			*c[0].add(lb + 1) = acc[1][0].lane0();
			*c[1].add(lb + 1) = acc[1][0].lane1();
			*c[2].add(lb + 1) = acc[1][1].lane0();
			*c[3].add(lb + 1) = acc[1][1].lane1();
			*c[0].add(lb + 2) = acc[2][0].lane0();
			*c[1].add(lb + 2) = acc[2][0].lane1();
			*c[2].add(lb + 2) = acc[2][1].lane0();
			*c[3].add(lb + 2) = acc[2][1].lane1();
			*c[0].add(lb + 3) = acc[3][0].lane0();
			*c[1].add(lb + 3) = acc[3][0].lane1();
			*c[2].add(lb + 3) = acc[3][1].lane0();
			*c[3].add(lb + 3) = acc[3][1].lane1();
		}
		for l in (0..r).rev() {
			let mut acc: [F64x2; 2] = [F64x2::pair(*c[0].add(l), *c[1].add(l)), F64x2::pair(*c[2].add(l), *c[3].add(l))];
			if !unit {
				let d = F64x2::splat(*ap.add(l * acs + l));
				acc[0] = acc[0].mul(d);
				acc[1] = acc[1].mul(d);
			}
			let mut i = l;
			while i > 0 {
				i -= 1;
				let t0 = F64x2::pair(*c[0].add(i), *c[1].add(i));
				let t1 = F64x2::pair(*c[2].add(i), *c[3].add(i));
				let av = F64x2::splat(*ap.add(l * acs + i));
				acc[0] = acc[0].add(av.mul(t0));
				acc[1] = acc[1].add(av.mul(t1));
			}
			*c[0].add(l) = acc[0].lane0();
			*c[1].add(l) = acc[0].lane1();
			*c[2].add(l) = acc[1].lane0();
			*c[3].add(l) = acc[1].lane1();
		}
	}
}

/// One right-hand side of [`dtrmm_left_t_k`], scalar, the same
/// per-element sequence.
fn dtrmm_left_t1(m: usize, a: &[f64], acs: usize, upper: bool, unit: bool, x: &mut [f64]) {
	if upper {
		for l in 0..m {
			let mut acc = x[l];
			if !unit {
				acc *= a[l * acs + l];
			}
			for i in l + 1..m {
				acc += a[l * acs + i] * x[i];
			}
			x[l] = acc;
		}
	} else {
		let mut l = m;
		while l > 0 {
			l -= 1;
			let mut acc = x[l];
			if !unit {
				acc *= a[l * acs + l];
			}
			let mut i = l;
			while i > 0 {
				i -= 1;
				acc += a[l * acs + i] * x[i];
			}
			x[l] = acc;
		}
	}
}

/// The K1 candidate for the left side: `B ← α·A·B` on the packing
/// layer, in place. The right panel is packed from B's rows at the
/// start of each k-block, before any row of that block is overwritten
/// (k-blocks ascend for upper, so only rows below the block have been
/// written; they descend for lower, so only rows above it have), forward
/// for an upper A and reversed for a lower one so the tile's ascending
/// loop reproduces the sweep's descending l; the MR×MR
/// diagonal block of each row block runs a scalar routine that starts
/// each element at `B[i,j]·A[i,i]` and adds the in-block terms in the
/// sweep's order; the tiles then continue from the stored partial. Row
/// and column tails keep the sweep. Per-element rounding sequence
/// identical to the shipping sweep (the diagonal product, the off-
/// diagonal terms ascending for upper / descending for lower, then
/// ×α): bit-for-bit interchangeable, tested. Raced against the shipping
/// sweep and against OpenBLAS compiled to wasm (`bench/README.md`).
#[allow(clippy::too_many_arguments)]
pub fn dtrmm_left_packed(upper: bool, unit: bool, alpha: f64, m: usize, n: usize, a: &[f64], acs: usize, b: &mut [f64], bcs: usize, work: &mut [f64]) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let need = dtrmm_left_packed_work(m, n);
	assert!(work.len() >= need, "dtrmm_left_packed: workspace shorter than dtrmm_left_packed_work");
	if m == 0 || n == 0 {
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	// column tail: the plain sweep per column (independent of the rest)
	for j in n4..n {
		let col = &mut b[j * bcs..j * bcs + m];
		dtrmv_k(m, a, acs, upper, unit, col);
		if alpha != 1.0 {
			dscal(alpha, col);
		}
	}
	if n4 == 0 {
		return;
	}
	let at = |i: usize, l: usize| a[l * acs + i];
	// the row tail: rows m_mr..m, stored terms only, in the sweep's
	// order. Upper reads only tail rows, so it runs after the main loop,
	// whose last panel must still see the original tail rows; lower reads
	// the rows above it, so it runs first, before any of them is
	// overwritten (no one reads tail rows in the lower case)
	let row_tail = |b: &mut [f64]| {
		for j in 0..n4 {
			// each tail row reads only rows on its far side (below for
			// upper, above for lower), so the rows are visited away from
			// that side and read before they are written
			for step in 0..m - m_mr {
				let r = if upper { m_mr + step } else { m - 1 - step };
				let bj = &b[j * bcs..j * bcs + m];
				let mut acc = if unit { bj[r] } else { bj[r] * at(r, r) };
				if upper {
					for (l, &v) in bj.iter().enumerate().skip(r + 1) {
						acc += at(r, l) * v;
					}
				} else {
					for (l, &v) in bj.iter().enumerate().take(r).rev() {
						acc += at(r, l) * v;
					}
				}
				b[j * bcs + r] = acc;
			}
		}
	};
	if !upper {
		row_tail(b);
	}
	if m_mr > 0 {
		let kcm = KC.min(m);
		let (bp, ap) = work[..need].split_at_mut(kcm * n4);
		// k-blocks: ascending for upper, descending (and reversed inside)
		// for lower, so every element's off-block terms arrive in the
		// sweep's order
		let nblk = m.div_ceil(KC);
		for bi in 0..nblk {
			let blk = if upper { bi } else { nblk - 1 - bi };
			let pc = blk * KC;
			let kc = KC.min(m - pc);
			// pack the right panel from B's rows pc..pc+kc, none of which has
			// been overwritten yet (upper: only rows below pc have; lower:
			// only rows from pc+kc up); per 4-column group, k-major ×4
			for jg in 0..n4 / 4 {
				let dst = &mut bp[jg * kc * 4..(jg + 1) * kc * 4];
				for t in 0..kc {
					let l = if upper { pc + t } else { pc + kc - 1 - t };
					for u in 0..4 {
						dst[t * 4 + u] = b[(jg * 4 + u) * bcs + l];
					}
				}
			}
			let mut ic = 0usize;
			while ic < m_mr {
				let mc = MC.min(m_mr - ic);
				for g in 0..mc / MR {
					let r0 = ic + g * MR;
					if pc <= r0 && r0 < pc + kc {
						// the diagonal MR×MR block, in the k-block that holds it:
						// each element starts at B·A_ii and takes the in-block
						// terms in the sweep's order, four columns per lane pair
						// read straight from the panel
						for jg in 0..n4 / 4 {
							unsafe {
								diag_block_rows(upper, unit, r0, pc, kc, a, acs, bp.as_ptr().add(jg * kc * 4), b.as_mut_ptr().add(jg * 4 * bcs), bcs);
							}
						}
					}
					// the panel range this row block multiplies: the l strictly
					// past its diagonal block, within this k-block
					let (t0, t1) = if upper {
						if pc + kc <= r0 + MR {
							continue;
						}
						((r0 + MR).max(pc) - pc, kc)
					} else {
						if pc >= r0 {
							continue;
						}
						// reversed: t ↔ l = pc + kc − 1 − t; l < r0 ⇔ t > pc + kc − 1 − r0
						((pc + kc).saturating_sub(r0), kc)
					};
					if t0 >= t1 {
						continue;
					}
					let base = g * kc * MR;
					for t in t0..t1 {
						let l = if upper { pc + t } else { pc + kc - 1 - t };
						let src = l * acs + r0;
						ap[base + t * MR..base + t * MR + MR].copy_from_slice(&a[src..src + MR]);
					}
					for jg in 0..n4 / 4 {
						unsafe {
							tile_4x4_packed_bcast(
								t1 - t0,
								ap.as_ptr().add(base + t0 * MR),
								bp.as_ptr().add(jg * kc * 4 + t0 * 4),
								1.0,
								false,
								b.as_mut_ptr().add(jg * 4 * bcs + r0),
								bcs,
							);
						}
					}
				}
				ic += mc;
			}
		}
	}
	if upper {
		row_tail(b);
	}
	if alpha != 1.0 {
		for j in 0..n4 {
			dscal(alpha, &mut b[j * bcs..j * bcs + m]);
		}
	}
}

/// Elements of scratch [`dtrmm_left_packed`] needs: the right panel and
/// the left panel; zero when the packed loops have nothing to do.
pub fn dtrmm_left_packed_work(m: usize, n: usize) -> usize {
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	if m_mr == 0 || n4 == 0 {
		0
	} else {
		let kc = KC.min(m);
		kc * n4 + MC.min(m_mr) * kc
	}
}

/// The diagonal MR×MR block of the packed left trmm for one 4-column
/// group: row `i` of the block starts at `B[i,·]·A[i,i]` (or `B[i,·]`
/// when unit) and adds `A[i,l]·B[l,·]` for the in-block `l` in the
/// sweep's order (ascending past the diagonal for upper, descending
/// before it for lower), the four columns as two lane pairs read from
/// the packed right panel (`bp`, k-major ×4, forward for upper and
/// reversed for lower); the result is stored into B's four columns.
/// # Safety
/// `bp` must hold the panel of this k-block for this column group; `b`
/// points at `B[0, j]` for the group's first column with stride `bcs`
/// and rows `r0..r0 + MR` in bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn diag_block_rows(upper: bool, unit: bool, r0: usize, pc: usize, kc: usize, a: &[f64], acs: usize, bp: *const f64, b: *mut f64, bcs: usize) {
	use crate::lanes::F64x2;
	let row = |l: usize| -> usize { if upper { l - pc } else { pc + kc - 1 - l } };
	for i in r0..r0 + MR {
		let t = row(i);
		let (mut lo, mut hi) = (F64x2::load(bp.add(t * 4)), F64x2::load(bp.add(t * 4 + 2)));
		if !unit {
			let d = F64x2::splat(a[i * acs + i]);
			lo = lo.mul(d);
			hi = hi.mul(d);
		}
		let mut step = |l: usize| {
			let ail = F64x2::splat(a[l * acs + i]);
			let tl = row(l);
			lo = lo.add(ail.mul(F64x2::load(bp.add(tl * 4))));
			hi = hi.add(ail.mul(F64x2::load(bp.add(tl * 4 + 2))));
		};
		if upper {
			for l in i + 1..r0 + MR {
				step(l);
			}
		} else {
			for l in (r0..i).rev() {
				step(l);
			}
		}
		*b.add(i) = lo.lane0();
		*b.add(bcs + i) = lo.lane1();
		*b.add(2 * bcs + i) = hi.lane0();
		*b.add(3 * bcs + i) = hi.lane1();
	}
}

// ---- the K1 candidates for the right side ----------------------------
//
// Both run the sweep's column grouping exactly (start-aligned groups of
// four for the ascending direction, end-aligned for the descending one,
// the same tail columns) and the sweep's per-element term order; what
// changes is that the out-of-group terms, a gemm over the already-final
// or original columns, run through the packed tile instead of a column
// fan-out.

/// B's rows packed as the left panels of every k-block over `n`
/// columns: per k-block, per MR-row block, k-major ×MR. Returns the
/// panel base of a k-block.
pub(crate) fn right_panel_base(blk: usize, n: usize, m_mr: usize) -> usize {
	let mut base = 0usize;
	for b in 0..blk {
		base += KC.min(n - b * KC) * m_mr;
	}
	base
}

/// The K1 candidate for the right side: `B ← α·B·A` on the packing
/// layer. B's packed-row copy serves as the left panels (m·n elements,
/// a cost of order mn against mn² of arithmetic); each group of four
/// columns first takes its in-group terms with the sweep's own column
/// code (the diagonal product and the in-group adds in the sweep's
/// order), then the out-of-group terms arrive through the tile in
/// ascending k from the right panel `α·A[k, group]`; the `m % MR` tail
/// rows take the same out-of-group terms from a copy of their original
/// values; the tail columns run the sweep. Per-element rounding
/// sequence identical to the shipping sweep: bit-for-bit
/// interchangeable, tested. Raced against the shipping sweep and
/// against OpenBLAS compiled to wasm (`bench/README.md`).
#[allow(clippy::too_many_arguments)]
pub fn dtrmm_right_packed(upper: bool, unit: bool, alpha: f64, m: usize, n: usize, a: &[f64], acs: usize, b: &mut [f64], bcs: usize, work: &mut [f64]) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	let need = dtrmm_right_packed_work(m, n);
	assert!(work.len() >= need, "dtrmm_right_packed: workspace shorter than dtrmm_right_packed_work");
	if m == 0 || n == 0 {
		return;
	}
	let at = |row: usize, col: usize| a[col * acs + row];
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	let mt = m - m_mr;
	// the sweep's grouping: upper descends over end-aligned groups with
	// the tail columns 0..off first in the matrix (done last); lower
	// ascends over start-aligned groups with the tail at the end
	let off = if upper { n % 4 } else { 0 };
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref = |idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const f64, m) };
	// tail columns: the plain column sweep (they read only each other)
	let do_col = |j: usize| {
		let diag = if unit { 1.0 } else { at(j, j) };
		dscal(alpha * diag, col_mut(j));
		let (lo, hi) = if upper { (0, j) } else { (j + 1, n) };
		for k in lo..hi {
			daxpy_k(alpha * at(k, j), col_ref(k), col_mut(j));
		}
	};
	// tail columns run last, as the sweep does: the groups read them as
	// original values, and they read only each other
	let tail_cols = || {
		if upper {
			for j in (0..off).rev() {
				do_col(j);
			}
		} else {
			for j in n4..n {
				do_col(j);
			}
		}
	};
	if n4 == 0 {
		tail_cols();
		return;
	}
	let (lp, rest) = work[..need].split_at_mut(m_mr * n);
	let (tail, rp) = rest.split_at_mut(mt * n);
	// the original B: rows < m_mr into the left panels of every k-block,
	// the tail rows into their own copy (column-major, m - m_mr rows)
	let nblk = n.div_ceil(KC);
	for blk in 0..nblk {
		let pc = blk * KC;
		let kc = KC.min(n - pc);
		let base = right_panel_base(blk, n, m_mr);
		for g in 0..m_mr / MR {
			let r0 = g * MR;
			for t in 0..kc {
				let src = (pc + t) * bcs + r0;
				lp[base + g * kc * MR + t * MR..base + g * kc * MR + t * MR + MR].copy_from_slice(&b[src..src + MR]);
			}
		}
	}
	for k in 0..n {
		tail[k * mt..k * mt + mt].copy_from_slice(&b[k * bcs + m_mr..k * bcs + m]);
	}
	// in-group pass, the sweep's own order (the diagonal product first)
	for q in 0..n4 / 4 {
		let gs = off + q * 4;
		if upper {
			for tc in (gs..gs + 4).rev() {
				let diag = if unit { 1.0 } else { at(tc, tc) };
				dscal(alpha * diag, col_mut(tc));
				for k in gs..tc {
					daxpy_k(alpha * at(k, tc), col_ref(k), col_mut(tc));
				}
			}
		} else {
			for tc in gs..gs + 4 {
				let diag = if unit { 1.0 } else { at(tc, tc) };
				dscal(alpha * diag, col_mut(tc));
				for k in tc + 1..gs + 4 {
					daxpy_k(alpha * at(k, tc), col_ref(k), col_mut(tc));
				}
			}
		}
	}
	// out-of-group terms, ascending k: the tile over the packed rows, the
	// tail rows by column-axpy from their copy
	for blk in 0..nblk {
		let pc = blk * KC;
		let kc = KC.min(n - pc);
		let base = right_panel_base(blk, n, m_mr);
		for q in 0..n4 / 4 {
			let gs = off + q * 4;
			// this group's out-of-group k within the k-block
			let (k0, k1) = if upper { (pc, gs.min(pc + kc)) } else { ((gs + 4).max(pc), pc + kc) };
			if k0 >= k1 {
				continue;
			}
			let dst = &mut rp[..(k1 - k0) * 4];
			for (t, k) in (k0..k1).enumerate() {
				for u in 0..4 {
					dst[t * 4 + u] = alpha * at(k, gs + u);
				}
			}
			for g in 0..m_mr / MR {
				unsafe {
					tile_4x4_packed_bcast(
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
						daxpy_k(dst[t * 4 + u], &tail[k * mt..k * mt + mt], &mut b[(gs + u) * bcs + m_mr..(gs + u) * bcs + m]);
					}
				}
			}
		}
	}
	tail_cols();
}

/// Elements of scratch [`dtrmm_right_packed`] needs: B's packed rows,
/// the tail rows' copy and one right panel.
pub fn dtrmm_right_packed_work(m: usize, n: usize) -> usize {
	let n4 = n - n % 4;
	if m == 0 || n4 == 0 {
		0
	} else {
		let m_mr = m - m % MR;
		m_mr * n + (m - m_mr) * n + KC.min(n) * 4
	}
}


/// The shipping left shape before K1, the lockstep column sweep: kept
/// as the race's loser and the bit-identity reference of
/// [`dtrmm_left_packed`]; exported so the tests and the benchmarks can
/// call it. Asserts rather than returning errors.
#[allow(clippy::too_many_arguments)]
pub fn dtrmm_left_sweep(upper: bool, unit: bool, alpha: f64, m: usize, n: usize, a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {
	dtrmm_left_k(alpha, m, n, a, lda, upper, unit, b, ldb);
}

/// The shipping right shape before K1, the grouped column sweep: kept
/// as the race's loser and the bit-identity reference of
/// [`dtrmm_right_packed`]; exported so the tests and the benchmarks can
/// call it. Asserts rather than returning errors.
#[allow(clippy::too_many_arguments)]
pub fn dtrmm_right_sweep(upper: bool, unit: bool, alpha: f64, m: usize, n: usize, a: &[f64], lda: usize, b: &mut [f64], ldb: usize) {
	dtrmm_right_k(alpha, m, n, a, lda, upper, unit, b, ldb, Trans::NoTrans);
}
