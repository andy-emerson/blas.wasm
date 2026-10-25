//! `zgemm` — complex matrix multiplication: C ← αAB + βC.
//!
//! Implementation: size-dispatched:
//! 4-column fused column-zaxpy (`kernels::zaxpy4`) below 1 MB of A,
//! BLIS-style packed-panel 2×4 complex register tile at and above
//! (two runner draws unanimous at every measured size: 1.08–1.15× at
//! 256³ to 1.33× at 1024³). The packed tile is the c64 register
//! geometry the col4-era doc recorded as a non-starter — packing
//! removes the strided k-walk that made it one. The plain
//! zgemv-per-column loop stays as `zgemm_colaxpy`, the raced-and-
//! bit-checked reference; all shapes are bit-for-bit identical,
//! per-element sequence βC then ascending k. Transposed or conjugated
//! operands are copied (conjugation an exact sign flip) into the
//! workspace by `L3::transpose_into` and ride the same shapes.

use super::check_mat;
use crate::{Error, Trans};
use crate::c64::C64;
use crate::kernels::zaxpy4;
use crate::L2::zgemv_k;

// Packed-panel sizes and the routing threshold, shared by the
// dispatcher, the packed shape and the workspace query.
const KC: usize = 256;
const MC: usize = 64;
const MR: usize = 2;
const PACKED_MIN_A_BYTES: usize = 1 << 20; // 1 MB

/// `C ← α·op(A)·op(B) + βC` with `op(A)` `m×k` and `op(B)` `k×n`, `C`
/// `m×n`, each at its own leading dimension. `op(X)` is `X`, `Xᵀ` or
/// `Xᴴ` per `transa`/`transb`. `work` is scratch of at least
/// [`zgemm_work`]`(transa, transb, m, n, k)` elements; it may be empty
/// when the query says zero.
///
/// A transposed operand is transposed (and conjugated) into the
/// workspace first and the product then runs the no-transpose shapes,
/// so every flag combination equals the unflagged call on an explicit
/// copy, bit for bit.
///
/// Errors: a leading dimension below its row count (arguments 8, 10,
/// 13); `a`, `b` or `c` shorter than its extent (arguments 7, 9, 12);
/// `work` shorter than the query (argument 14).
#[allow(clippy::too_many_arguments)]
pub fn zgemm(transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: C64, a: &[C64], lda: usize, b: &[C64], ldb: usize, beta: C64, c: &mut [C64], ldc: usize, work: &mut [C64]) -> Result<(), Error> {
	const R: &str = "zgemm";
	let (ar, ac) = if transa == Trans::NoTrans { (m, k) } else { (k, m) };
	let (br, bc) = if transb == Trans::NoTrans { (k, n) } else { (n, k) };
	crate::error::mat(R, 7, 8, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 9, 10, b.len(), br, bc, ldb)?;
	crate::error::mat(R, 12, 13, c.len(), m, n, ldc)?;
	crate::error::work(R, 14, work.len(), zgemm_work(transa, transb, m, n, k))?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if k == 0 {
		// a product of empty operands: C ← βC, the same per-column
		// scaling every shape applies before accumulating
		for j in 0..n {
			crate::L2::zscale_y(beta, &mut c[j * ldc..j * ldc + m]);
		}
		return Ok(());
	}
	// op(A) and op(B) are materialized in the workspace ahead of the
	// no-transpose shapes, so every flag combination is bit-identical
	// to the unflagged call on an explicit copy (tested); conjugation
	// is an exact sign flip applied in the copy
	let (work, a, lda) = if transa == Trans::NoTrans {
		(work, a, lda)
	} else {
		let (ta, rest) = work.split_at_mut(m * k);
		if transa == Trans::ConjTrans {
			super::transpose_into(a, lda, k, m, ta, |x| x.conj());
		} else {
			super::transpose_into(a, lda, k, m, ta, |x| x);
		}
		(rest, &*ta, m)
	};
	let (work, b, ldb) = if transb == Trans::NoTrans {
		(work, b, ldb)
	} else {
		let (tb, rest) = work.split_at_mut(k * n);
		if transb == Trans::ConjTrans {
			super::transpose_into(b, ldb, n, k, tb, |x| x.conj());
		} else {
			super::transpose_into(b, ldb, n, k, tb, |x| x);
		}
		(rest, &*tb, k)
	};
	zgemm_k(alpha, m, n, k, a, lda, b, ldb, beta, c, ldc, work);
	Ok(())
}

/// Elements of scratch [`zgemm`] needs for these flags and dimensions:
/// `m·k` for a transposed `A` and `k·n` for a transposed `B` (the
/// copies), plus [`zgemm_packed_work`] when the dispatcher routes to
/// the packed shape; zero for an unflagged call below that size.
pub fn zgemm_work(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> usize {
	let ta = if transa == Trans::NoTrans { 0 } else { m * k };
	let tb = if transb == Trans::NoTrans { 0 } else { k * n };
	let packed = if packed_route(m, k) { zgemm_packed_work(m, n, k) } else { 0 };
	ta + tb + packed
}

/// Elements of scratch [`zgemm_packed`] needs for these dimensions:
/// the expanded B panel (`KC` rows of the `n − n % 4` packed columns,
/// 2 complexes per element) and the A panel (`MC × KC`); zero when
/// the packed loops have nothing to do.
pub fn zgemm_packed_work(m: usize, n: usize, k: usize) -> usize {
	let n4 = n - n % 4;
	if k == 0 || n4 == 0 || m < MR {
		0
	} else {
		KC * n4 * 2 + MC * KC
	}
}

/// Whether the dispatcher routes `m×k` `A` to the packed-panel shape.
#[inline]
pub(crate) fn packed_route(m: usize, k: usize) -> bool {
	m * k * 16 >= PACKED_MIN_A_BYTES
}

/// C ← αAB + βC. A is m×k, B is k×n, C is m×n; each matrix has its
/// own column stride.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zgemm_k(
	alpha: C64,
	m: usize,
	n: usize,
	k: usize,
	a: &[C64],
	acs: usize,
	b: &[C64],
	bcs: usize,
	beta: C64,
	c: &mut [C64],
	ccs: usize,
	work: &mut [C64],
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	// packed-gemm race (two runner draws, unanimous at every
	// measured size): packed wins 1.08–1.15x at 256³ (A = 1 MB, the
	// smallest measured point) up to 1.33x at 1024³. Below 1 MB of A is
	// unmeasured — col4 keeps it.
	if m * k * 16 >= PACKED_MIN_A_BYTES {
		return zgemm_packed(alpha, m, n, k, a, acs, b, bcs, beta, c, ccs, work);
	}
	let mut j = 0usize;
	while j + 4 <= n {
		{
			let cp = c.as_mut_ptr();
			for u in 0..4 {
				let col =
					unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs), m) };
				crate::L2::zscale_y(beta, col);
			}
			for l in 0..k {
				let t = [
					alpha * b[j * bcs + l],
					alpha * b[(j + 1) * bcs + l],
					alpha * b[(j + 2) * bcs + l],
					alpha * b[(j + 3) * bcs + l],
				];
				unsafe {
					zaxpy4(
						a.as_ptr().add(l * acs),
						t,
						cp.add(j * ccs),
						cp.add((j + 1) * ccs),
						cp.add((j + 2) * ccs),
						cp.add((j + 3) * ccs),
						m,
					);
				}
			}
		}
		j += 4;
	}
	while j < n {
		zgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate: packed-panel zgemm — the
/// BLIS/Goto structure of `dgemm_packed` at complex register geometry.
/// This is the c64 register tile the col4 module doc recorded as "a
/// different register geometry, not a mechanical port": 2 complexes ×
/// 4 columns of C held in 8 F64x2 accumulators, fed from packed panels
/// so the k-walk is sequential (the strided-A weakness that killed the
/// idea of an unpacked complex tile never arises). α is folded into
/// the B panel at pack time (same single α·b rounding as col4's `t`),
/// stored pre-expanded as `[re,re]` / [−im,im] lane pairs so the kernel
/// loads its multipliers instead of rebuilding them per step.
///
/// Per-element sequence: βC first (shared `zscale_y`, applied to whole
/// columns up front), then ascending-k adds of the proven bit-exact
/// product form `av·vre + swap(av)·vim` — identical to col4/colaxpy;
/// bit-for-bit interchangeable, tested. Tails (m%2 rows, n%4 columns)
/// fall back to the plain per-column sequences.
#[allow(clippy::too_many_arguments)]
pub fn zgemm_packed(
	alpha: C64,
	m: usize,
	n: usize,
	k: usize,
	a: &[C64],
	acs: usize,
	b: &[C64],
	bcs: usize,
	beta: C64,
	c: &mut [C64],
	ccs: usize,
	work: &mut [C64],
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let need = zgemm_packed_work(m, n, k);
	assert!(work.len() >= need, "zgemm_packed: workspace shorter than zgemm_packed_work");
	// Ap = MC·KC complexes = 256 KB (L2-resident); one Bp column-group
	// strip = KC·16 f64 = 32 KB (L1-resident).
	if k == 0 {
		for j in 0..n {
			crate::L2::zscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	// β once per element, before any accumulation — same order as the
	// per-column passes
	for j in 0..n4 {
		crate::L2::zscale_y(beta, &mut c[j * ccs..j * ccs + m]);
	}
	if n4 > 0 && m_mr > 0 {
		// bp holds, per (l, column u): [t.re, t.re, −t.im, t.im] with
		// t = α·b already rounded — 4 f64, stored as two C64 of the
		// caller's workspace (`repr(C)` pairs, read by the kernel as
		// f64 lanes). Every element the kernel reads is packed first,
		// so the slice's prior contents never reach C (tested with a
		// NaN fill).
		let (bp, ap) = work[..need].split_at_mut(KC * n4 * 2);
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			for jg in 0..n4 / 4 {
				for l in 0..kc {
					for u in 0..4 {
						let t = alpha * b[(jg * 4 + u) * bcs + pc + l];
						let o = jg * kc * 8 + l * 8 + u * 2;
						bp[o] = C64::new(t.re, t.re);
						bp[o + 1] = C64::new(-t.im, t.im);
					}
				}
			}
			let mut ic = 0usize;
			while ic < m_mr {
				let mc = MC.min(m_mr - ic); // both multiples of MR
				for g in 0..mc / MR {
					let base = g * kc * MR;
					for l in 0..kc {
						let src = (pc + l) * acs + ic + g * MR;
						ap[base + l * MR..base + l * MR + MR]
							.copy_from_slice(&a[src..src + MR]);
					}
				}
				for jg in 0..n4 / 4 {
					for g in 0..mc / MR {
						unsafe {
							ztile_2x4_packed(
								kc,
								ap.as_ptr().add(g * kc * MR) as *const f64,
								bp.as_ptr().add(jg * kc * 8) as *const f64,
								c.as_mut_ptr().add(jg * 4 * ccs + ic + g * MR),
								ccs,
							);
						}
					}
				}
				ic += mc;
			}
			pc += kc;
		}
	}
	// row tail (β already applied above): plain zaxpy per k step
	if m_mr < m {
		for j in 0..n4 {
			for l in 0..k {
				crate::L1::zaxpy_k(
					alpha * b[j * bcs + l],
					&a[l * acs + m_mr..l * acs + m],
					&mut c[j * ccs + m_mr..j * ccs + m],
				);
			}
		}
	}
	// column tail: plain zgemv columns (handle their own β)
	for j in n4..n {
		zgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// One 2×4 complex microkernel step over PACKED panels: 2 complexes
/// (2 F64x2 accumulator rows) × 4 columns. `ap` walks 2 complexes
/// (4 f64) per k step; `bp` walks 4 pre-expanded multiplier pairs
/// (16 f64) per k step. Product form and per-element rounding order
/// identical to `zaxpy4` (`c + (av·vre + swap(av)·vim)`).
/// # Safety
/// `ap` valid for `kc*4` f64, `bp` for `kc*16` f64; `cp` points at
/// `C[i, j]` (stride `ccs` complexes) with 2 rows and 4 columns in
/// bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn ztile_2x4_packed(
	kc: usize,
	ap: *const f64,
	bp: *const f64,
	cp: *mut C64,
	ccs: usize,
) {
	use crate::lanes::F64x2;
	let cpf = cp as *mut f64;
	let mut acc = [[F64x2::splat(0.0); 2]; 4];
	for (u, au) in acc.iter_mut().enumerate() {
		au[0] = F64x2::load(cpf.add(u * ccs * 2));
		au[1] = F64x2::load(cpf.add(u * ccs * 2 + 2));
	}
	for l in 0..kc {
		let a0 = F64x2::load(ap.add(l * 4));
		let a1 = F64x2::load(ap.add(l * 4 + 2));
		let a0s = a0.swap();
		let a1s = a1.swap();
		for (u, au) in acc.iter_mut().enumerate() {
			let vre = F64x2::load(bp.add(l * 16 + u * 4));
			let vim = F64x2::load(bp.add(l * 16 + u * 4 + 2));
			au[0] = au[0].add(a0.mul(vre).add(a0s.mul(vim)));
			au[1] = au[1].add(a1.mul(vre).add(a1s.mul(vim)));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cpf.add(u * ccs * 2));
		au[1].store(cpf.add(u * ccs * 2 + 2));
	}
}

/// The plain column-zaxpy shape (zgemv per column) — kept as the
/// reference the fused shape is bit-checked against.
#[allow(clippy::too_many_arguments)]
pub fn zgemm_colaxpy(
	alpha: C64,
	m: usize,
	n: usize,
	k: usize,
	a: &[C64],
	acs: usize,
	b: &[C64],
	bcs: usize,
	beta: C64,
	c: &mut [C64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		zgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}
