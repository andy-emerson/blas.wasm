//! `ztrmm` — triangular matrix multiply, in place: B ← αAB (left) or
//! B ← αBA (right), A triangular (complex).
//!
//! Implementation: the `dtrmm` shapes verbatim with complex scalars —
//! left side walks four B columns in lockstep through the `ztrmv`
//! column sweep sharing each A column read (`kernels::zaxpy4`); right
//! side runs the in-group contributions in the original elimination
//! order, then fans out-of-group source columns via one stream each.
//! Same reorder disclosure as f64: LOWER-right keeps the plain
//! sweep's per-element order bit-for-bit; UPPER-right's in-group adds
//! precede the out-of-group adds — deterministic, documented, and
//! mirrored by the tests' scalar replay. The transposed and
//! conjugate-transposed forms route between a workspace copy (left) and
//! the no-copy accessor kernel (right) per the race (`bench/README.md`).

use super::super::check_mat;
use crate::{Diag, Error, Side, Trans, Uplo};
use crate::c64::C64;
use crate::kernels::zaxpy4;
use crate::L1::{zaxpy_k, zscal};
use crate::L2::ztrmv_k;

/// `B ← α·op(A)·B` ([`Side::Left`]) or `B ← α·B·op(A)` ([`Side::Right`])
/// for a triangular `A`, the triangle named by `uplo` stored, with a
/// unit diagonal when `diag` says so; `A` is `m×m` on the left and `n×n`
/// on the right, `B` is `m×n`, each at its own leading dimension.
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ` per `transa`. `work` is scratch of at
/// least [`ztrmm_work`]`(side, transa, m, n)` elements.
///
/// The transposed forms are defined as the no-transpose form on an
/// explicit transpose (conjugate transpose), and two
/// candidates compute them, [`ztrmm_copy`] and [`ztrmm_stream`],
/// both bit-identical to that definition; which one this routine routes
/// to per side is the race's verdict (`bench/README.md`; decision D3 in `DESIGN.md`).
///
/// Errors: a leading dimension below its row count (arguments 9, 11);
/// `a` or `b` shorter than its extent (arguments 8, 10); `work` shorter
/// than the query (argument 12).
#[allow(clippy::too_many_arguments)]
pub fn ztrmm(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: C64, a: &[C64], lda: usize, b: &mut [C64], ldb: usize, work: &mut [C64]) -> Result<(), Error> {
	const R: &str = "ztrmm";
	let na = if side == Side::Left { m } else { n };
	crate::error::mat(R, 8, 9, a.len(), na, na, lda)?;
	crate::error::mat(R, 10, 11, b.len(), m, n, ldb)?;
	crate::error::work(R, 12, work.len(), ztrmm_work(side, transa, m, n))?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if alpha == C64::ZERO {
		// reference: B ← 0 without reading A or B
		for j in 0..n {
			b[j * ldb..j * ldb + m].fill(C64::ZERO);
		}
		return Ok(());
	}
	match (side, transa) {
		(Side::Left, Trans::NoTrans) => ztrmm_left_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb),
		(Side::Right, Trans::NoTrans) => ztrmm_right_k(alpha, m, n, a, lda, uplo.is_upper(), diag.is_unit(), b, ldb, Trans::NoTrans),
		(Side::Left, _) if LEFT_T_STREAM => ztrmm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		(Side::Right, _) if RIGHT_T_STREAM => ztrmm_stream(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb),
		_ => ztrmm_copy(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, work),
	}
	Ok(())
}

/// Elements of scratch [`ztrmm`] needs for these flags and dimensions:
/// zero for the no-transpose forms and for a transposed form routed to
/// the streaming candidate; the `na×na` copy ([`ztrmm_copy_work`])
/// where the copy candidate is routed.
pub fn ztrmm_work(side: Side, transa: Trans, m: usize, n: usize) -> usize {
	if m == 0 || n == 0 || transa == Trans::NoTrans {
		return 0;
	}
	let stream = if side == Side::Left { LEFT_T_STREAM } else { RIGHT_T_STREAM };
	if stream { 0 } else { ztrmm_copy_work(side, m, n) }
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

/// B ← αAB. A is m×m triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ztrmm_left_k(
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
		let bp = b.as_mut_ptr();
		let cols: [*mut C64; 4] = [
			unsafe { bp.add(j * bcs) },
			unsafe { bp.add((j + 1) * bcs) },
			unsafe { bp.add((j + 2) * bcs) },
			unsafe { bp.add((j + 3) * bcs) },
		];
		// the ztrmv column sweep, four x vectors in lockstep sharing
		// each A column read
		if upper {
			for l in 0..m {
				let t = unsafe {
					[*cols[0].add(l), *cols[1].add(l), *cols[2].add(l), *cols[3].add(l)]
				};
				unsafe {
					zaxpy4(a.as_ptr().add(l * acs), t, cols[0], cols[1], cols[2], cols[3], l);
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
				if !unit {
					let d = a[l * acs + l];
					for (u, cu) in cols.iter().enumerate() {
						unsafe { *cu.add(l) = t[u] * d };
					}
				}
			}
		}
		if alpha != C64::ONE {
			for u in 0..4 {
				zscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		j += 4;
	}
	while j < n {
		let col = &mut b[j * bcs..j * bcs + m];
		ztrmv_k(m, a, acs, upper, unit, col);
		if alpha != C64::ONE {
			zscal(alpha, col);
		}
		j += 1;
	}
}

/// B ← αBA. A is n×n triangular, B is m×n.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ztrmm_right_k(
	alpha: C64,
	m: usize,
	n: usize,
	a: &[C64],
	acs: usize,
	upper: bool,
	unit: bool,
	b: &mut [C64],
	bcs: usize,
	op: Trans,
) {
	check_mat(a.len(), n, n, acs);
	check_mat(b.len(), m, n, bcs);
	// `upper` and the accessor describe op(A): for a transposed op the
	// stored triangle is the other one and the element roles swap
	let at = |row: usize, col: usize| -> C64 {
		match op {
			Trans::NoTrans => a[col * acs + row],
			Trans::Trans => a[row * acs + col],
			Trans::ConjTrans => a[row * acs + col].conj(),
		}
	};
	// result column j = α·Σ_{k in triangle} a`[k,j]`·B_orig[:,k]; column
	// order (descending for upper, ascending for lower) touches each
	// original column before anything overwrites it. Columns j and k
	// never alias (k ≠ j always), so the raw-pointer views are
	// disjoint.
	let bp = b.as_mut_ptr();
	let col_mut = |idx: usize| unsafe { core::slice::from_raw_parts_mut(bp.add(idx * bcs), m) };
	let col_ref =
		|idx: usize| unsafe { core::slice::from_raw_parts(bp.add(idx * bcs) as *const C64, m) };
	let do_col = |j: usize| {
		let diag = if unit { C64::ONE } else { at(j, j) };
		zscal(alpha * diag, col_mut(j));
		let (lo, hi) = if upper { (0, j) } else { (j + 1, n) };
		for k in lo..hi {
			zaxpy_k(alpha * at(k, j), col_ref(k), col_mut(j));
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
		let r = n % 4;
		let mut gs = n;
		while gs >= r + 4 {
			gs -= 4;
			// in-group first, original descending order (sources still
			// original when read)
			for tc in (gs..gs + 4).rev() {
				let diag = if unit { C64::ONE } else { at(tc, tc) };
				zscal(alpha * diag, col_mut(tc));
				for k in gs..tc {
					zaxpy_k(alpha * at(k, tc), col_ref(k), col_mut(tc));
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
				let diag = if unit { C64::ONE } else { at(tc, tc) };
				zscal(alpha * diag, col_mut(tc));
				for k in tc + 1..gs + 4 {
					zaxpy_k(alpha * at(k, tc), col_ref(k), col_mut(tc));
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
/// is transposed (and conjugated) into `work` and the no-transpose
/// kernel runs on the copy with the triangle flipped, so the result is
/// the explicit-copy definition itself. `work` needs
/// [`ztrmm_copy_work`]`(side, m, n)` elements. Exported so the
/// tests and the benchmarks can race it; it asserts rather than
/// returning errors, and `transa` must not be `NoTrans`.
#[allow(clippy::too_many_arguments)]
pub fn ztrmm_copy(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: C64, a: &[C64], lda: usize, b: &mut [C64], ldb: usize, work: &mut [C64]) {
	assert!(transa != Trans::NoTrans, "ztrmm_copy: a transposed form only");
	let na = if side == Side::Left { m } else { n };
	check_mat(a.len(), na, na, lda);
	assert!(work.len() >= na * na, "ztrmm_copy: workspace shorter than ztrmm_copy_work");
	let ta = &mut work[..na * na];
		if transa == Trans::ConjTrans {
			super::super::transpose_into(a, lda, na, na, ta, |x| x.conj());
		} else {
			super::super::transpose_into(a, lda, na, na, ta, |x| x);
		}
	let upper = !uplo.is_upper();
	match side {
		Side::Left => ztrmm_left_k(alpha, m, n, ta, na.max(1), upper, diag.is_unit(), b, ldb),
		Side::Right => ztrmm_right_k(alpha, m, n, ta, na.max(1), upper, diag.is_unit(), b, ldb, Trans::NoTrans),
	}
}

/// Elements of scratch [`ztrmm_copy`] needs: the `na×na` copy of
/// op(A), `na` being `m` on the left and `n` on the right.
pub fn ztrmm_copy_work(side: Side, m: usize, n: usize) -> usize {
	let na = if side == Side::Left { m } else { n };
	na * na
}

/// Candidate for the transposed forms (decision D3 in `DESIGN.md`; the race in `bench/README.md`): no
/// copy; on the right the no-transpose kernel reads A through a
/// transposed accessor (it only ever reads A as scalars), on the left
/// `ztrmm_left_t_k` reduces down the stored columns. Bit-identical
/// to [`ztrmm_copy`] (tested). Exported so the tests and the
/// benchmarks can race it; it asserts rather than returning errors, and
/// `transa` must not be `NoTrans`.
#[allow(clippy::too_many_arguments)]
pub fn ztrmm_stream(side: Side, uplo: Uplo, transa: Trans, diag: Diag, m: usize, n: usize, alpha: C64, a: &[C64], lda: usize, b: &mut [C64], ldb: usize) {
	assert!(transa != Trans::NoTrans, "ztrmm_stream: a transposed form only");
	let upper = !uplo.is_upper();
	match side {
		Side::Left => ztrmm_left_t_k(alpha, m, n, a, lda, upper, transa == Trans::ConjTrans, diag.is_unit(), b, ldb),
		Side::Right => ztrmm_right_k(alpha, m, n, a, lda, upper, diag.is_unit(), b, ldb, transa),
	}
}

/// B ← α·op(A)·B with op(A) = Aᵀ (`conj`: Aᴴ), computed from the
/// stored A without a copy: `upper` names op(A)'s triangle (stored A
/// holds the other). Each element's arithmetic sequence is the lockstep
/// ztrmv sweep's — the diagonal product first, then the products added one at a
/// time down the stored column (ascending for upper, descending for
/// lower), then the α scaling — in the scalar complex product form, which is
/// bit-identical to the lane form (`c64.rs`), so the result equals
/// [`ztrmm_left_k`] on an explicit transpose (tested). Four right-hand
/// sides go in lockstep and rows two at a time, so one step loads two A
/// values and four B values for eight complex multiply-adds; op(A)'s
/// row l is stored column l, read contiguously. Conjugation is applied
/// to each A value as it is read (an exact sign flip).
#[allow(clippy::too_many_arguments)]
pub(crate) fn ztrmm_left_t_k(
	alpha: C64,
	m: usize,
	n: usize,
	a: &[C64],
	acs: usize,
	upper: bool,
	conj: bool,
	unit: bool,
	b: &mut [C64],
	bcs: usize,
) {
	check_mat(a.len(), m, m, acs);
	check_mat(b.len(), m, n, bcs);
	let cj = |v: C64| if conj { v.conj() } else { v };
	let ap = a.as_ptr();
	let mut j = 0usize;
	while j + 4 <= n {
		let bp = b.as_mut_ptr();
		let c: [*mut C64; 4] = [
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
				let mut lb = 0usize;
				while lb + 2 <= m {
					let mut acc: [[C64; 4]; 2] = [[*c[0].add(lb), *c[1].add(lb), *c[2].add(lb), *c[3].add(lb)], [*c[0].add(lb + 1), *c[1].add(lb + 1), *c[2].add(lb + 1), *c[3].add(lb + 1)]];
					let x = acc;
					if !unit {
						let d0 = cj(*ap.add(lb * acs + lb));
						let d1 = cj(*ap.add((lb + 1) * acs + lb + 1));
						for v in acc[0].iter_mut() {
							*v = *v * d0;
						}
						for v in acc[1].iter_mut() {
							*v = *v * d1;
						}
					}
					let a01 = cj(*ap.add(lb * acs + lb + 1));
					for u in 0..4 {
						acc[0][u] = acc[0][u] + a01 * x[1][u];
					}
					for i in lb + 2..m {
						let a0 = cj(*ap.add(lb * acs + i));
						let a1 = cj(*ap.add((lb + 1) * acs + i));
						for u in 0..4 {
							let t = *c[u].add(i);
							acc[0][u] = acc[0][u] + a0 * t;
							acc[1][u] = acc[1][u] + a1 * t;
						}
					}
					for u in 0..4 {
						*c[u].add(lb) = acc[0][u];
						*c[u].add(lb + 1) = acc[1][u];
					}
					lb += 2;
				}
				if lb < m {
					let l = lb;
					let mut acc = [*c[0].add(l), *c[1].add(l), *c[2].add(l), *c[3].add(l)];
					if !unit {
						let d = cj(*ap.add(l * acs + l));
						for v in acc.iter_mut() {
							*v = *v * d;
						}
					}
					for i in l + 1..m {
						let av = cj(*ap.add(l * acs + i));
						for (u, v) in acc.iter_mut().enumerate() {
							*v = *v + av * *c[u].add(i);
						}
					}
					for (u, v) in acc.iter().enumerate() {
						*c[u].add(l) = *v;
					}
				}
			} else {
				let r = m % 2;
				let mut lb = m;
				while lb >= r + 2 {
					lb -= 2;
					let mut acc: [[C64; 4]; 2] = [[*c[0].add(lb), *c[1].add(lb), *c[2].add(lb), *c[3].add(lb)], [*c[0].add(lb + 1), *c[1].add(lb + 1), *c[2].add(lb + 1), *c[3].add(lb + 1)]];
					let x = acc;
					if !unit {
						let d0 = cj(*ap.add(lb * acs + lb));
						let d1 = cj(*ap.add((lb + 1) * acs + lb + 1));
						for v in acc[0].iter_mut() {
							*v = *v * d0;
						}
						for v in acc[1].iter_mut() {
							*v = *v * d1;
						}
					}
					let a10 = cj(*ap.add((lb + 1) * acs + lb));
					for u in 0..4 {
						acc[1][u] = acc[1][u] + a10 * x[0][u];
					}
					let mut i = lb;
					while i > 0 {
						i -= 1;
						let a0 = cj(*ap.add(lb * acs + i));
						let a1 = cj(*ap.add((lb + 1) * acs + i));
						for u in 0..4 {
							let t = *c[u].add(i);
							acc[0][u] = acc[0][u] + a0 * t;
							acc[1][u] = acc[1][u] + a1 * t;
						}
					}
					for u in 0..4 {
						*c[u].add(lb) = acc[0][u];
						*c[u].add(lb + 1) = acc[1][u];
					}
				}
				if r == 1 {
					let l = 0usize;
					let mut acc = [*c[0].add(l), *c[1].add(l), *c[2].add(l), *c[3].add(l)];
					if !unit {
						let d = cj(*ap.add(l * acs + l));
						for v in acc.iter_mut() {
							*v = *v * d;
						}
					}
					let mut i = l;
					while i > 0 {
						i -= 1;
						let av = cj(*ap.add(l * acs + i));
						for (u, v) in acc.iter_mut().enumerate() {
							*v = *v + av * *c[u].add(i);
						}
					}
					for (u, v) in acc.iter().enumerate() {
						*c[u].add(l) = *v;
					}
				}
			}
		}
		if alpha != C64::ONE {
			for u in 0..4 {
				zscal(alpha, &mut b[(j + u) * bcs..(j + u) * bcs + m]);
			}
		}
		j += 4;
	}
	while j < n {
		let x = &mut b[j * bcs..j * bcs + m];
		if upper {
			for l in 0..m {
				let mut acc = x[l];
				if !unit {
				acc = acc * cj(a[l * acs + l]);
			}
				for i in l + 1..m {
				acc = acc + cj(a[l * acs + i]) * x[i];
			}
				x[l] = acc;
			}
		} else {
			let mut l = m;
		while l > 0 {
			l -= 1;
				let mut acc = x[l];
				if !unit {
				acc = acc * cj(a[l * acs + l]);
			}
				let mut i = l;
			while i > 0 {
				i -= 1;
				acc = acc + cj(a[l * acs + i]) * x[i];
			}
				x[l] = acc;
			}
		}
		if alpha != C64::ONE {
			zscal(alpha, x);
		}
		j += 1;
	}
}
