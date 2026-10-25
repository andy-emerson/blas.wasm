//! `cgemm` — complex matrix multiplication: C ← αAB + βC.
//!
//! Implementation: size-dispatched:
//! 4-column fused column-caxpy (`kernels::caxpy4`) below 8 MB of A,
//! BLIS-style packed-panel 4×4 complex register tile at and above —
//! the only zone where the packed shape measured a win (+12% at
//! 1024³, two runner draws unanimous; 256³–512³ and deep-K were
//! noise, so col4 keeps them). The plain cgemv-per-column loop stays
//! as `cgemm_colaxpy`, the raced-and-bit-checked reference; all
//! shapes are bit-for-bit identical, per-element sequence βC then
//! ascending k. Transposed or conjugated operands
//! (conjugation an exact sign flip) are packed from a transposed read
//! above the packed threshold (`cgemm_packed_op`) and copied into the
//! workspace below it (`cgemm_copy_op`), per the race.

use super::check_mat;
use crate::{Error, Trans};
use crate::c32::C32;
use crate::kernels::caxpy4;
use crate::L2::cgemv_k;

// Packed-panel sizes and the routing threshold, shared by the
// dispatcher, the packed shape and the workspace query.
const KC: usize = 256;
const MC: usize = 128;
const MR: usize = 4;
const PACKED_MIN_A_BYTES: usize = 8 << 20; // 8 MB

/// `C ← α·op(A)·op(B) + βC` with `op(A)` `m×k` and `op(B)` `k×n`, `C`
/// `m×n`, each at its own leading dimension. `op(X)` is `X`, `Xᵀ` or
/// `Xᴴ` per `transa`/`transb`. `work` is scratch of at least
/// [`cgemm_work`]`(transa, transb, m, n, k)` elements; it may be empty
/// when the query says zero.
///
/// A transposed operand is packed straight from a transposed (and conjugated)
/// read above the packed threshold ([`cgemm_packed_op`]) and transposed
/// into the workspace ahead of the register-tile shape below it, so every
/// flag combination equals the unflagged call on an explicit copy, bit
/// for bit.
///
/// Errors: a leading dimension below its stored row count (arguments 8,
/// 10, 13); `a`, `b` or `c` shorter than its extent (arguments 7, 9, 12);
/// `work` shorter than the query (argument 14).
#[allow(clippy::too_many_arguments)]
pub fn cgemm(transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: C32, a: &[C32], lda: usize, b: &[C32], ldb: usize, beta: C32, c: &mut [C32], ldc: usize, work: &mut [C32]) -> Result<(), Error> {
	const R: &str = "cgemm";
	let (ar, ac) = if transa == Trans::NoTrans { (m, k) } else { (k, m) };
	let (br, bc) = if transb == Trans::NoTrans { (k, n) } else { (n, k) };
	crate::error::mat(R, 7, 8, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 9, 10, b.len(), br, bc, ldb)?;
	crate::error::mat(R, 12, 13, c.len(), m, n, ldc)?;
	crate::error::work(R, 14, work.len(), cgemm_work(transa, transb, m, n, k))?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if k == 0 || alpha == C32::ZERO {
		// reference: C ← βC without reading A or B (β = 1 leaves C as
		// is, β = 0 writes zeros), the same per-column scaling every
		// shape applies before accumulating
		for j in 0..n {
			crate::L2::cscale_y(beta, &mut c[j * ldc..j * ldc + m]);
		}
		return Ok(());
	}
	// Above the packed threshold a transposed A is packed straight from a
	// transposed read (`cgemm_packed_op`, the race's winner there:
	// `bench/README.md`); a transposed B alone is copied into the
	// workspace first (`cgemm_copy_op`), which beat the transposed pack
	// on every c32 row in both draws (0.90–0.99), as does every form below
	// the threshold. Both candidates are bit-identical to the unflagged
	// call on an explicit copy (tested); conjugation is an exact sign flip
	// in either.
	if transa != Trans::NoTrans && packed_route(m, k) {
		cgemm_packed_op(transa, transb, alpha, m, n, k, a, lda, b, ldb, beta, c, ldc, work);
		return Ok(());
	}
	cgemm_copy_op(transa, transb, alpha, m, n, k, a, lda, b, ldb, beta, c, ldc, work);
	Ok(())
}

/// Elements of scratch [`cgemm`] needs for these flags and dimensions:
/// above the packed threshold [`cgemm_packed_work`] (unflagged) or
/// [`cgemm_packed_op_work`] (a transposed operand); below it the
/// copies, `m·k` for a transposed `A` and `k·n` for a transposed `B`,
/// so zero for an unflagged call there.
pub fn cgemm_work(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> usize {
	if m == 0 || n == 0 || k == 0 {
		return 0; // the quick returns compute nothing
	}
	if packed_route(m, k) && transa != Trans::NoTrans {
		cgemm_packed_op_work(transa, transb, m, n, k)
	} else {
		cgemm_copy_op_work(transa, transb, m, n, k)
	}
}

/// Elements of scratch [`cgemm_packed`] needs for these dimensions:
/// the expanded B panel (`KC` rows of the `n − n % 4` packed columns,
/// 4 complexes per element) and the A panel (`MC × KC`); zero when
/// the packed loops have nothing to do.
pub fn cgemm_packed_work(m: usize, n: usize, k: usize) -> usize {
	let n4 = n - n % 4;
	if k == 0 || n4 == 0 || m < MR {
		0
	} else {
		KC * n4 * 4 + MC * KC
	}
}

/// Whether the dispatcher routes `m×k` `A` to the packed-panel shape.
#[inline]
pub(crate) fn packed_route(m: usize, k: usize) -> bool {
	m * k * 8 >= PACKED_MIN_A_BYTES
}

/// C ← αAB + βC. A is m×k, B is k×n, C is m×n; each matrix has its
/// own column stride.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cgemm_k(
	alpha: C32,
	m: usize,
	n: usize,
	k: usize,
	a: &[C32],
	acs: usize,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
	work: &mut [C32],
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	// packed-gemm race (two runner draws): packed wins only
	// at 1024³ (+12%, A = 8 MB, unanimous); 256³/512³/deep-K measured
	// noise. Routed at the measured win only — the (2 MB, 8 MB) gap is
	// unmeasured and col4 keeps it.
	if m * k * 8 >= PACKED_MIN_A_BYTES {
		return cgemm_packed(alpha, m, n, k, a, acs, b, bcs, beta, c, ccs, work);
	}
	let mut j = 0usize;
	while j + 4 <= n {
		{
			let cp = c.as_mut_ptr();
			for u in 0..4 {
				let col =
					unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs), m) };
				crate::L2::cscale_y(beta, col);
			}
			for l in 0..k {
				let t = [
					alpha * b[j * bcs + l],
					alpha * b[(j + 1) * bcs + l],
					alpha * b[(j + 2) * bcs + l],
					alpha * b[(j + 3) * bcs + l],
				];
				unsafe {
					caxpy4(
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
		cgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// The plain column-caxpy shape (cgemv per column) — kept as the
/// reference the fused shape is bit-checked against.
#[allow(clippy::too_many_arguments)]
pub fn cgemm_colaxpy(
	alpha: C32,
	m: usize,
	n: usize,
	k: usize,
	a: &[C32],
	acs: usize,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		cgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// Tuning-campaign candidate: packed-panel cgemm — the
/// `zgemm_packed` structure at c32 pair geometry: 4 complexes (2 F32x4
/// accumulator rows) × 4 columns of C, fed from packed panels. α folds
/// into the B panel at pack time (same single α·b rounding as col4's
/// `t`), stored pre-expanded as [re×4] / [−im,im,−im,im] lanes so the
/// kernel loads its multipliers. Per-element sequence: βC first
/// (shared `cscale_y`), then ascending-k adds of the proven product
/// form `av·vre + swap_pairs(av)·vim` — identical to col4/colaxpy;
/// bit-for-bit interchangeable, tested. Tails (m%4 rows, n%4 columns)
/// fall back to the plain per-column sequences.
#[allow(clippy::too_many_arguments)]
pub fn cgemm_packed(
	alpha: C32,
	m: usize,
	n: usize,
	k: usize,
	a: &[C32],
	acs: usize,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
	work: &mut [C32],
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let need = cgemm_packed_work(m, n, k);
	assert!(work.len() >= need, "cgemm_packed: workspace shorter than cgemm_packed_work");
	// Ap = MC·KC complexes = 256 KB (L2-resident); one Bp column-group
	// strip = KC·32 f32 = 32 KB (L1-resident).
	if k == 0 {
		for j in 0..n {
			crate::L2::cscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	for j in 0..n4 {
		crate::L2::cscale_y(beta, &mut c[j * ccs..j * ccs + m]);
	}
	if n4 > 0 && m_mr > 0 {
		// bp holds, per (l, column u): [t.re ×4] then [−t.im, t.im,
		// −t.im, t.im] with t = α·b already rounded — 8 f32, stored as
		// four C32 of the caller's workspace (`repr(C)` pairs, read by
		// the kernel as f32 lanes). Every element the kernel reads is
		// packed first, so the slice's prior contents never reach C
		// (tested with a NaN fill).
		let (bp, ap) = work[..need].split_at_mut(KC * n4 * 4);
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			for jg in 0..n4 / 4 {
				for l in 0..kc {
					for u in 0..4 {
						let t = alpha * b[(jg * 4 + u) * bcs + pc + l];
						let o = jg * kc * 16 + l * 16 + u * 4;
						bp[o] = C32::new(t.re, t.re);
						bp[o + 1] = C32::new(t.re, t.re);
						bp[o + 2] = C32::new(-t.im, t.im);
						bp[o + 3] = C32::new(-t.im, t.im);
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
							ctile_4x4_packed(
								kc,
								ap.as_ptr().add(g * kc * MR) as *const f32,
								bp.as_ptr().add(jg * kc * 16) as *const f32,
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
	// row tail (β already applied above): plain caxpy per k step
	if m_mr < m {
		for j in 0..n4 {
			for l in 0..k {
				crate::L1::caxpy_k(
					alpha * b[j * bcs + l],
					&a[l * acs + m_mr..l * acs + m],
					&mut c[j * ccs + m_mr..j * ccs + m],
				);
			}
		}
	}
	// column tail: plain cgemv columns (handle their own β)
	for j in n4..n {
		cgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// One 4×4 complex microkernel step over PACKED panels: 4 complexes
/// (2 F32x4 accumulator rows, 2 complexes per register) × 4 columns.
/// `ap` walks 4 complexes (8 f32) per k step; `bp` walks 4
/// pre-expanded multiplier pairs (32 f32) per k step. Product form and
/// per-element rounding order identical to `caxpy4`
/// (`c + (av·vre + swap_pairs(av)·vim)`).
/// # Safety
/// `ap` valid for `kc*8` f32, `bp` for `kc*32` f32; `cp` points at
/// `C[i, j]` (stride `ccs` complexes) with 4 rows and 4 columns in
/// bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn ctile_4x4_packed(
	kc: usize,
	ap: *const f32,
	bp: *const f32,
	cp: *mut C32,
	ccs: usize,
) {
	use crate::lanes::F32x4;
	let cpf = cp as *mut f32;
	let mut acc = [[F32x4::splat(0.0); 2]; 4];
	for (u, au) in acc.iter_mut().enumerate() {
		au[0] = F32x4::load(cpf.add(u * ccs * 2));
		au[1] = F32x4::load(cpf.add(u * ccs * 2 + 4));
	}
	for l in 0..kc {
		let a0 = F32x4::load(ap.add(l * 8));
		let a1 = F32x4::load(ap.add(l * 8 + 4));
		let a0s = a0.swap_pairs();
		let a1s = a1.swap_pairs();
		for (u, au) in acc.iter_mut().enumerate() {
			let vre = F32x4::load(bp.add(l * 32 + u * 8));
			let vim = F32x4::load(bp.add(l * 32 + u * 8 + 4));
			au[0] = au[0].add(a0.mul(vre).add(a0s.mul(vim)));
			au[1] = au[1].add(a1.mul(vre).add(a1s.mul(vim)));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cpf.add(u * ccs * 2));
		au[1].store(cpf.add(u * ccs * 2 + 4));
	}
}

/// The other candidate for the transposed forms (`STATUS.md`, decision
/// D3): a transposed operand is transposed (and conjugated) into `work`
/// with `transpose_into` and the product then runs the size-dispatched
/// no-transpose shapes on the copy, so every flag combination is the
/// explicit-copy definition itself. The netlib wrapper routes it below
/// the packed threshold, where the two candidates coincide; above it the
/// race chose [`cgemm_packed_op`]. `work` needs [`cgemm_copy_op_work`]
/// elements. Exported so the benchmarks can race it; it asserts rather
/// than returning errors.
#[allow(clippy::too_many_arguments)]
pub fn cgemm_copy_op(
	transa: Trans,
	transb: Trans,
	alpha: C32,
	m: usize,
	n: usize,
	k: usize,
	a: &[C32],
	lda: usize,
	b: &[C32],
	ldb: usize,
	beta: C32,
	c: &mut [C32],
	ldc: usize,
	work: &mut [C32],
) {
	assert!(work.len() >= cgemm_copy_op_work(transa, transb, m, n, k), "cgemm_copy_op: workspace shorter than cgemm_copy_op_work");
	if m == 0 || n == 0 || k == 0 {
		// nothing to copy: C ← βC as every shape does before accumulating
		for j in 0..n {
			crate::L2::cscale_y(beta, &mut c[j * ldc..j * ldc + m]);
		}
		return;
	}
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
	cgemm_k(alpha, m, n, k, a, lda, b, ldb, beta, c, ldc, work);
}

/// Elements of scratch [`cgemm_copy_op`] needs: the copies (`m·k` for a
/// transposed `A`, `k·n` for a transposed `B`) plus [`cgemm_packed_work`]
/// when the dispatcher routes to the packed shape.
pub fn cgemm_copy_op_work(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> usize {
	if m == 0 || n == 0 || k == 0 {
		return 0;
	}
	let ta = if transa == Trans::NoTrans { 0 } else { m * k };
	let tb = if transb == Trans::NoTrans { 0 } else { k * n };
	let packed = if packed_route(m, k) { cgemm_packed_work(m, n, k) } else { 0 };
	ta + tb + packed
}

/// The transposed forms above the packed threshold (the race's winner
/// over a whole-operand copy, two runner draws: `bench/README.md`): the
/// packed-panel shape with `op(A)` and
/// `op(B)` packed straight from a transposed (and conjugated) read —
/// op(A)'s row is a stored column of A and op(B)'s packed quad is four
/// consecutive stored elements — so no whole-operand copy is made and
/// the microkernel is unchanged. The tails that read the operands
/// directly are kept bit-identical by small gathers into the workspace:
/// op(A)'s row tail (`m % MR` rows × k), op(B)'s column tail (k × `n %
/// 4`), and, for a transposed A, the column tail's first `m − m % MR`
/// rows ride a fourth packed column group padded with zeros into a `4 ×
/// (m − m % MR)` scratch block of C. `work` needs
/// [`cgemm_packed_op_work`] elements. Bit-identical to [`cgemm_packed`]
/// on an explicit transpose (tested); with both flags `NoTrans` it is
/// [`cgemm_packed`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn cgemm_packed_op(
	transa: Trans,
	transb: Trans,
	alpha: C32,
	m: usize,
	n: usize,
	k: usize,
	a: &[C32],
	acs: usize,
	b: &[C32],
	bcs: usize,
	beta: C32,
	c: &mut [C32],
	ccs: usize,
	work: &mut [C32],
) {
	let ta = transa != Trans::NoTrans;
	let tb = transb != Trans::NoTrans;
	let (ar, ac) = if ta { (k, m) } else { (m, k) };
	let (br, bc) = if tb { (n, k) } else { (k, n) };
	check_mat(a.len(), ar, ac, acs);
	check_mat(b.len(), br, bc, bcs);
	check_mat(c.len(), m, n, ccs);
	let lay = packed_op_layout(transa, transb, m, n, k);
	let need: usize = lay.iter().sum();
	assert!(work.len() >= need, "cgemm_packed_op: workspace shorter than cgemm_packed_op_work");
	if k == 0 {
		for j in 0..n {
			crate::L2::cscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	let mt = m - m_mr;
	let nt = n - n4;
	let pad = ta && nt > 0 && m_mr > 0;
	let bpn = if pad { n4 + 4 } else { n4 };
	// workspace: B panel | A panel | op(A) row tail | op(B) column tail | padded C block
	let (bp, rest) = work[..need].split_at_mut(lay[0]);
	let (ap, rest) = rest.split_at_mut(lay[1]);
	let (wa, rest) = rest.split_at_mut(lay[2]);
	let (wb, ct) = rest.split_at_mut(lay[3]);
	let cja = |v: C32| if transa == Trans::ConjTrans { v.conj() } else { v };
	let cjb = |v: C32| if transb == Trans::ConjTrans { v.conj() } else { v };
	// op(A)[i, l] and op(B)[l, j] as stored
	let ao = |i: usize, l: usize| if ta { cja(a[i * acs + l]) } else { a[l * acs + i] };
	let bo = |l: usize, j: usize| if tb { cjb(b[l * bcs + j]) } else { b[j * bcs + l] };
	if ta {
		for r in 0..mt {
			for l in 0..k {
				wa[l * mt + r] = ao(m_mr + r, l);
			}
		}
	}
	if tb && !pad {
		for jj in 0..nt {
			for l in 0..k {
				wb[jj * k + l] = bo(l, n4 + jj);
			}
		}
	}
	// β once per element, before any accumulation — same order as the
	// per-column passes; the padded block carries the scaled tail columns
	for j in 0..n4 {
		crate::L2::cscale_y(beta, &mut c[j * ccs..j * ccs + m]);
	}
	if pad {
		for jj in 0..nt {
			let col = &mut c[(n4 + jj) * ccs..(n4 + jj) * ccs + m];
			crate::L2::cscale_y(beta, col);
			ct[jj * m_mr..jj * m_mr + m_mr].copy_from_slice(&col[..m_mr]);
		}
	}
	if bpn > 0 && m_mr > 0 {
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			for jg in 0..bpn / 4 {
				for l in 0..kc {
					for u in 0..4 {
						let j = jg * 4 + u;
						let t = if j < n { alpha * bo(pc + l, j) } else { C32::ZERO };
						let o = jg * kc * 16 + l * 16 + u * 4;
						bp[o] = C32::new(t.re, t.re);
						bp[o + 1] = C32::new(t.re, t.re);
						bp[o + 2] = C32::new(-t.im, t.im);
						bp[o + 3] = C32::new(-t.im, t.im);
					}
				}
			}
			let mut ic = 0usize;
			while ic < m_mr {
				let mc = MC.min(m_mr - ic);
				for g in 0..mc / MR {
					let base = g * kc * MR;
					if ta {
						for r in 0..MR {
							for l in 0..kc {
								ap[base + l * MR + r] = ao(ic + g * MR + r, pc + l);
							}
						}
					} else {
						for l in 0..kc {
							let src = (pc + l) * acs + ic + g * MR;
							ap[base + l * MR..base + l * MR + MR].copy_from_slice(&a[src..src + MR]);
						}
					}
				}
				for jg in 0..n4 / 4 {
					for g in 0..mc / MR {
						unsafe {
							ctile_4x4_packed(
								kc,
								ap.as_ptr().add(g * kc * MR) as *const f32,
								bp.as_ptr().add(jg * kc * 16) as *const f32,
								c.as_mut_ptr().add(jg * 4 * ccs + ic + g * MR),
								ccs,
							);
						}
					}
				}
				if pad {
					for g in 0..mc / MR {
						unsafe {
							ctile_4x4_packed(
								kc,
								ap.as_ptr().add(g * kc * MR) as *const f32,
								bp.as_ptr().add((n4 / 4) * kc * 16) as *const f32,
								ct.as_mut_ptr().add(ic + g * MR),
								m_mr,
							);
						}
					}
				}
				ic += mc;
			}
			pc += kc;
		}
	}
	if pad {
		for jj in 0..nt {
			c[(n4 + jj) * ccs..(n4 + jj) * ccs + m_mr].copy_from_slice(&ct[jj * m_mr..jj * m_mr + m_mr]);
		}
	}
	// row tail × packed columns (β already applied): axpy per k step
	if mt > 0 {
		for j in 0..n4 {
			for l in 0..k {
				let col = if ta { &wa[l * mt..l * mt + mt] } else { &a[l * acs + m_mr..l * acs + m] };
				crate::L1::caxpy_k(alpha * bo(l, j), col, &mut c[j * ccs + m_mr..j * ccs + m]);
			}
		}
	}
	// column tail
	for jj in 0..nt {
		let j = n4 + jj;
		if ta {
			// β was applied with the padded block; the row tail's rows
			// take the same per-column axpy sequence
			if !pad {
				crate::L2::cscale_y(beta, &mut c[j * ccs..j * ccs + m]);
			}
			for l in 0..k {
				crate::L1::caxpy_k(alpha * bo(l, j), &wa[l * mt..l * mt + mt], &mut c[j * ccs + m_mr..j * ccs + m]);
			}
		} else {
			let bcol: &[C32] = if tb { &wb[jj * k..jj * k + k] } else { &b[j * bcs..j * bcs + k] };
			cgemv_k(alpha, m, k, a, acs, bcol, beta, &mut c[j * ccs..j * ccs + m]);
		}
	}
}

/// The workspace layout of [`cgemm_packed_op`]: B panel, A panel,
/// op(A)'s gathered row tail, op(B)'s gathered column tail, the padded
/// C block.
fn packed_op_layout(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> [usize; 5] {
	if k == 0 {
		return [0; 5];
	}
	let ta = transa != Trans::NoTrans;
	let tb = transb != Trans::NoTrans;
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	let mt = m - m_mr;
	let nt = n - n4;
	let pad = ta && nt > 0 && m_mr > 0;
	let bpn = if pad { n4 + 4 } else { n4 };
	let main = bpn > 0 && m_mr > 0;
	[
		if main { KC * bpn * 4 } else { 0 },
		if main { MC * KC } else { 0 },
		if ta { mt * k } else { 0 },
		if tb && !pad { k * nt } else { 0 },
		if pad { 4 * m_mr } else { 0 },
	]
}

/// Elements of scratch [`cgemm_packed_op`] needs for these flags and
/// dimensions; equal to [`cgemm_packed_work`] when both flags are
/// `NoTrans`.
pub fn cgemm_packed_op_work(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> usize {
	packed_op_layout(transa, transb, m, n, k).iter().sum()
}
