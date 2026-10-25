//! `sgemm` — matrix multiplication: C ← αAB + βC.
//!
//! Implementation: size-dispatched family: 8×4 register tile below ~3 MB of A,
//! BLIS-style packed-panel shape above (replaced the 4-column fused
//! stream: +3–5% at 1024³ and 512×4096×1024, two runner draws
//! unanimous; the tile keeps everything below, where packed measured
//! noise to −2%) — all bit-identical to the plain sgemv-per-column
//! reference, which is kept as `sgemm_colaxpy`; col4 stays as the
//! raced reference. f32 measurements: campaign record steps 10 and 14
//! (STATUS.md). Transposed operands are copied into the
//! workspace by `L3::transpose_into` and ride the same shapes.

use super::check_mat;
use crate::{Error, Trans};
use crate::kernels::saxpy4;
use crate::L2::sgemv_k;

// Packed-panel sizes and the routing threshold, shared by the
// dispatcher, the packed shape and the workspace query.
const KC: usize = 256;
const MC: usize = 128;
const MR: usize = 8;
const TILED_MAX_A_BYTES: usize = 3 << 20; // 3 MB

/// `C ← α·op(A)·op(B) + βC` with `op(A)` `m×k` and `op(B)` `k×n`, `C`
/// `m×n`, each at its own leading dimension. `op(X)` is `X`, `Xᵀ` or
/// `Xᴴ` per `transa`/`transb`. `work` is scratch of at least
/// [`sgemm_work`]`(transa, transb, m, n, k)` elements; it may be empty
/// when the query says zero.
///
/// A transposed operand is transposed into the
/// workspace first and the product then runs the no-transpose shapes,
/// so every flag combination equals the unflagged call on an explicit
/// copy, bit for bit; `ConjTrans` is `Trans` for a real matrix.
///
/// Errors: a leading dimension below its stored row count (arguments 8,
/// 10, 13); `a`, `b` or `c` shorter than its extent (arguments 7, 9, 12);
/// `work` shorter than the query (argument 14).
#[allow(clippy::too_many_arguments)]
pub fn sgemm(transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize, b: &[f32], ldb: usize, beta: f32, c: &mut [f32], ldc: usize, work: &mut [f32]) -> Result<(), Error> {
	const R: &str = "sgemm";
	let (ar, ac) = if transa == Trans::NoTrans { (m, k) } else { (k, m) };
	let (br, bc) = if transb == Trans::NoTrans { (k, n) } else { (n, k) };
	crate::error::mat(R, 7, 8, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 9, 10, b.len(), br, bc, ldb)?;
	crate::error::mat(R, 12, 13, c.len(), m, n, ldc)?;
	crate::error::work(R, 14, work.len(), sgemm_work(transa, transb, m, n, k))?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if k == 0 {
		// a product of empty operands: C ← βC, the same per-column
		// scaling every shape applies before accumulating
		for j in 0..n {
			crate::L2::sscale_y(beta, &mut c[j * ldc..j * ldc + m]);
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
		super::transpose_into(a, lda, k, m, ta, |x| x);
		(rest, &*ta, m)
	};
	let (work, b, ldb) = if transb == Trans::NoTrans {
		(work, b, ldb)
	} else {
		let (tb, rest) = work.split_at_mut(k * n);
		super::transpose_into(b, ldb, n, k, tb, |x| x);
		(rest, &*tb, k)
	};
	sgemm_k(alpha, m, n, k, a, lda, b, ldb, beta, c, ldc, work);
	Ok(())
}

/// Elements of scratch [`sgemm`] needs for these flags and dimensions:
/// `m·k` for a transposed `A` and `k·n` for a transposed `B` (the
/// copies), plus [`sgemm_packed_work`] when the dispatcher routes to
/// the packed shape; zero for an unflagged call below that size.
pub fn sgemm_work(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> usize {
	if m == 0 || n == 0 || k == 0 {
		return 0; // the quick returns compute nothing
	}
	let ta = if transa == Trans::NoTrans { 0 } else { m * k };
	let tb = if transb == Trans::NoTrans { 0 } else { k * n };
	let packed = if packed_route(m, k) { sgemm_packed_work(m, n, k) } else { 0 };
	ta + tb + packed
}

/// Elements of scratch [`sgemm_packed`] needs for these dimensions:
/// the B panel (`KC` rows of the `n − n % 4` packed columns) and the
/// A panel (`MC × KC`); zero when the packed loops have nothing to do.
pub fn sgemm_packed_work(m: usize, n: usize, k: usize) -> usize {
	let n4 = n - n % 4;
	if k == 0 || n4 == 0 || m < MR {
		0
	} else {
		KC * n4 + MC * KC
	}
}

/// Whether the dispatcher routes `m×k` `A` to the packed-panel shape.
#[inline]
pub(crate) fn packed_route(m: usize, k: usize) -> bool {
	m * k * 4 > TILED_MAX_A_BYTES
}

/// C ← αAB + βC. A is m×k, B is k×n, C is m×n; each matrix has its own
/// column stride.
///
/// Dispatches by A's byte size at the runner-raced f32 crossover (the
/// register tile wins while A rides the caches, the 4-column fused
/// stream wins above; two draws, campaign record step 10). All three shapes are
/// bit-for-bit identical, so the dispatch is invisible to results.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm_k(
	alpha: f32,
	m: usize,
	n: usize,
	k: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
	work: &mut [f32],
) {
	// f32 crossover from the campaign record step-10 runner draws (which rule over the
	// container, where col4 led from n=512): tiled wins unanimously
	// through n=512, n=768 splits within noise — the 8-row tile
	// survives LONGER in bytes than the f64 4-row tile's 1.5 MB.
	// Above the threshold, the packed-panel shape replaced col4
	// (packed-gemm race, two runner draws: +3–4% at 1024³,
	// +5% at 512x4096x1024, unanimous; 512³/256³ stay tiled, where
	// packed measured noise/−2%). col4 stays as the raced reference.
	if m * k * 4 <= TILED_MAX_A_BYTES {
		sgemm_tiled(alpha, m, n, k, a, acs, b, bcs, beta, c, ccs);
	} else {
		sgemm_packed(alpha, m, n, k, a, acs, b, bcs, beta, c, ccs, work);
	}
}

/// The original column-saxpy shape (sgemv per column) — kept as the
/// plain reference the tuned shapes are raced and bit-checked against.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_colaxpy(
	alpha: f32,
	m: usize,
	n: usize,
	k: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		sgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// Tuning-campaign shape (ported from f64): 8×4 register-tiled sgemm.
/// The column-saxpy sgemm re-reads and re-writes each C element once per
/// k step; this micro-kernel holds an 8-row × 4-column tile of C in 8
/// SIMD registers across the whole k loop — one C load and one C store
/// per tile, 32 FLOPs per 2 A-register loads. Rounding sequence per
/// element is IDENTICAL to the column-saxpy path (β first, then one
/// α·b rounding and one multiply-add per k, ascending), so the two are
/// bit-for-bit interchangeable — tested. Tails (m%8, n%4) fall back to
/// the sgemv path.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_tiled(
	alpha: f32,
	m: usize,
	n: usize,
	k: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		let mut i = 0usize;
		while i + 8 <= m {
			unsafe {
				tile_8x4(
					alpha,
					k,
					a.as_ptr().add(i),
					acs,
					b.as_ptr().add(j * bcs),
					bcs,
					beta,
					c.as_mut_ptr().add(j * ccs + i),
					ccs,
				);
			}
			i += 8;
		}
		// row tail for these four columns: same per-element sequence as
		// the sgemv path over the remaining rows
		if i < m {
			for jj in j..j + 4 {
				let seg = &mut c[jj * ccs + i..jj * ccs + m];
				crate::L2::sscale_y(beta, seg);
				for l in 0..k {
					crate::L1::saxpy_k(
						alpha * b[jj * bcs + l],
						&a[l * acs + i..l * acs + m],
						seg,
					);
				}
			}
		}
		j += 4;
	}
	// column tail: plain sgemv columns
	while j < n {
		sgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate 2: 4-column fused sgemm.
/// The 4×4 tile's k-loop walks A at column stride (TLB/prefetch-hostile
/// at large n — measured losing above n≈512); this shape instead
/// streams each A column SEQUENTIALLY, once per group of four C
/// columns — A traffic drops 4× vs column-saxpy while the four hot C
/// columns ride the near caches. Rounding sequence per element is
/// identical to the column-saxpy path — bit-for-bit interchangeable,
/// tested.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_col4(
	alpha: f32,
	m: usize,
	n: usize,
	k: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		{
			let cp = c.as_mut_ptr();
			for u in 0..4 {
				let col =
					unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs), m) };
				crate::L2::sscale_y(beta, col);
			}
			for l in 0..k {
				let t = [
					alpha * b[j * bcs + l],
					alpha * b[(j + 1) * bcs + l],
					alpha * b[(j + 2) * bcs + l],
					alpha * b[(j + 3) * bcs + l],
				];
				unsafe {
					saxpy4(
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
		sgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate 3: packed-panel sgemm —
/// BLIS/Goto structure around the same 8×4 microkernel math; see
/// `dgemm_packed` for the full rationale (deep-K motivation, cache
/// roles of the two panels, bit-compat argument). Rounding sequence
/// per element is IDENTICAL to the column-saxpy path — k-blocking
/// resumes each element's accumulation from its stored partial —
/// bit-for-bit interchangeable, tested.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_packed(
	alpha: f32,
	m: usize,
	n: usize,
	k: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
	work: &mut [f32],
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let need = sgemm_packed_work(m, n, k);
	assert!(work.len() >= need, "sgemm_packed: workspace shorter than sgemm_packed_work");
	// Ap = MC·KC·4 = 128 KB (L2-resident), one Bp strip = KC·4·4 = 4 KB
	if k == 0 {
		for j in 0..n {
			crate::L2::sscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	if n4 > 0 && m_mr > 0 {
		// the caller's workspace, split into the two panels; every
		// element the microkernel reads is packed first, so the
		// slice's prior contents never reach C (tested with a NaN fill)
		let (bp, ap) = work[..need].split_at_mut(KC * n4);
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			for jg in 0..n4 / 4 {
				let dst = &mut bp[jg * kc * 4..(jg + 1) * kc * 4];
				let j = jg * 4;
				for l in 0..kc {
					for u in 0..4 {
						dst[l * 4 + u] = b[(j + u) * bcs + pc + l];
					}
				}
			}
			let first = pc == 0;
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
							tile_8x4_packed(
								alpha,
								kc,
								ap.as_ptr().add(g * kc * MR),
								bp.as_ptr().add(jg * kc * 4),
								beta,
								first,
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
	if m_mr < m {
		for j in 0..n4 {
			let seg = &mut c[j * ccs + m_mr..j * ccs + m];
			crate::L2::sscale_y(beta, seg);
			for l in 0..k {
				crate::L1::saxpy_k(alpha * b[j * bcs + l], &a[l * acs + m_mr..l * acs + m], seg);
			}
		}
	}
	for j in n4..n {
		sgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// One 8×4 microkernel step over PACKED panels: `ap` walks MR=8
/// contiguous row values per k step, `bp` walks 4 contiguous column
/// values per k step. Math and per-element rounding order identical to
/// [`tile_8x4`]; `first` selects β-application vs continuing from the
/// stored partial.
/// # Safety
/// `ap` must be valid for `kc*8` f32s, `bp` for `kc*4`; `cp` points at
/// `C[i, j]` (stride ccs) with 8 rows and 4 columns in bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_8x4_packed(
	alpha: f32,
	kc: usize,
	ap: *const f32,
	bp: *const f32,
	beta: f32,
	first: bool,
	cp: *mut f32,
	ccs: usize,
) {
	use crate::lanes::F32x4;
	let zero = F32x4::splat(0.0);
	let mut acc = [[zero; 2]; 4];
	if first {
		if beta != 0.0 {
			let vb = F32x4::splat(beta);
			for (u, au) in acc.iter_mut().enumerate() {
				au[0] = F32x4::load(cp.add(u * ccs)).mul(vb);
				au[1] = F32x4::load(cp.add(u * ccs + 4)).mul(vb);
			}
		}
	} else {
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = F32x4::load(cp.add(u * ccs));
			au[1] = F32x4::load(cp.add(u * ccs + 4));
		}
	}
	for l in 0..kc {
		let a0 = F32x4::load(ap.add(l * 8));
		let a1 = F32x4::load(ap.add(l * 8 + 4));
		for (u, au) in acc.iter_mut().enumerate() {
			let t = F32x4::splat(alpha * *bp.add(l * 4 + u));
			au[0] = au[0].add(a0.mul(t));
			au[1] = au[1].add(a1.mul(t));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cp.add(u * ccs));
		au[1].store(cp.add(u * ccs + 4));
	}
}

/// One 8×4 tile: rows i..i+8 (two f32x4 registers) × columns j..j+4 —
/// the f64 4×4 tile with the same register count at twice the lane
/// width.
/// # Safety
/// `ap` points at `A[i, 0]` (stride acs, k columns), `bp` at `B[0, j]`
/// (stride bcs), `cp` at `C[i, j]` (stride ccs); 8 rows and 4 B/C
/// columns must be in bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_8x4(
	alpha: f32,
	k: usize,
	ap: *const f32,
	acs: usize,
	bp: *const f32,
	bcs: usize,
	beta: f32,
	cp: *mut f32,
	ccs: usize,
) {
	use crate::lanes::F32x4;
	let zero = F32x4::splat(0.0);
	let vb = F32x4::splat(beta);
	let mut acc = [[zero; 2]; 4];
	if beta != 0.0 {
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = F32x4::load(cp.add(u * ccs)).mul(vb);
			au[1] = F32x4::load(cp.add(u * ccs + 4)).mul(vb);
		}
	}
	for l in 0..k {
		let a0 = F32x4::load(ap.add(l * acs));
		let a1 = F32x4::load(ap.add(l * acs + 4));
		for (u, au) in acc.iter_mut().enumerate() {
			let t = F32x4::splat(alpha * *bp.add(u * bcs + l));
			au[0] = au[0].add(a0.mul(t));
			au[1] = au[1].add(a1.mul(t));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cp.add(u * ccs));
		au[1].store(cp.add(u * ccs + 4));
	}
}

/// Candidate for the transposed forms (`STATUS.md`, decision D3; the
/// netlib wrapper routes the other candidate, a whole-operand copy, or
/// this one per the race): the packed-panel shape with `op(A)` and
/// `op(B)` packed straight from a transposed read — op(A)'s row is a
/// stored column of A and op(B)'s packed quad is four consecutive stored
/// elements — so no whole-operand copy is made and the microkernel is
/// unchanged. The three tails that read the operands directly are kept
/// bit-identical by small gathers into the workspace: op(A)'s row tail
/// (`m % MR` rows × k), op(B)'s column tail (k × `n % 4`), and, for a
/// transposed A, the column tail's first `m − m % MR` rows ride a fourth
/// packed column group padded with zeros into a `4 × (m − m % MR)`
/// scratch block of C. `work` needs [`sgemm_packed_op_work`] elements.
/// Bit-identical to [`sgemm_packed`] on an explicit transpose (tested);
/// with both flags `NoTrans` it is [`sgemm_packed`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn sgemm_packed_op(
	transa: Trans,
	transb: Trans,
	alpha: f32,
	m: usize,
	n: usize,
	k: usize,
	a: &[f32],
	acs: usize,
	b: &[f32],
	bcs: usize,
	beta: f32,
	c: &mut [f32],
	ccs: usize,
	work: &mut [f32],
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
	assert!(work.len() >= need, "sgemm_packed_op: workspace shorter than sgemm_packed_op_work");
	if k == 0 {
		for j in 0..n {
			crate::L2::sscale_y(beta, &mut c[j * ccs..j * ccs + m]);
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
	// op(B)[l, j] as stored
	let bo = |l: usize, j: usize| if tb { b[l * bcs + j] } else { b[j * bcs + l] };
	if ta {
		// op(A)[m_mr.., :] gathered as an mt×k matrix at leading dimension mt
		for r in 0..mt {
			let src = &a[(m_mr + r) * acs..(m_mr + r) * acs + k];
			for (l, &v) in src.iter().enumerate() {
				wa[l * mt + r] = v;
			}
		}
	}
	if tb && !pad {
		for jj in 0..nt {
			for l in 0..k {
				wb[jj * k + l] = b[l * bcs + n4 + jj];
			}
		}
	}
	if pad {
		for jj in 0..nt {
			ct[jj * m_mr..jj * m_mr + m_mr].copy_from_slice(&c[(n4 + jj) * ccs..(n4 + jj) * ccs + m_mr]);
		}
	}
	if bpn > 0 && m_mr > 0 {
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			for jg in 0..n4 / 4 {
				let dst = &mut bp[jg * kc * 4..(jg + 1) * kc * 4];
				let j = jg * 4;
				if tb {
					for l in 0..kc {
						dst[l * 4..l * 4 + 4].copy_from_slice(&b[(pc + l) * bcs + j..(pc + l) * bcs + j + 4]);
					}
				} else {
					for l in 0..kc {
						for u in 0..4 {
							dst[l * 4 + u] = b[(j + u) * bcs + pc + l];
						}
					}
				}
			}
			if pad {
				let dst = &mut bp[(n4 / 4) * kc * 4..(n4 / 4 + 1) * kc * 4];
				for l in 0..kc {
					for u in 0..4 {
						dst[l * 4 + u] = if u < nt { bo(pc + l, n4 + u) } else { 0.0 };
					}
				}
			}
			let first = pc == 0;
			let mut ic = 0usize;
			while ic < m_mr {
				let mc = MC.min(m_mr - ic);
				for g in 0..mc / MR {
					let base = g * kc * MR;
					if ta {
						for r in 0..MR {
							let src = &a[(ic + g * MR + r) * acs + pc..(ic + g * MR + r) * acs + pc + kc];
							for (l, &v) in src.iter().enumerate() {
								ap[base + l * MR + r] = v;
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
							tile_8x4_packed(
								alpha,
								kc,
								ap.as_ptr().add(g * kc * MR),
								bp.as_ptr().add(jg * kc * 4),
								beta,
								first,
								c.as_mut_ptr().add(jg * 4 * ccs + ic + g * MR),
								ccs,
							);
						}
					}
				}
				if pad {
					for g in 0..mc / MR {
						unsafe {
							tile_8x4_packed(
								alpha,
								kc,
								ap.as_ptr().add(g * kc * MR),
								bp.as_ptr().add((n4 / 4) * kc * 4),
								beta,
								first,
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
	// row tail × packed columns: per-column axpy over full k
	if mt > 0 {
		for j in 0..n4 {
			let seg = &mut c[j * ccs + m_mr..j * ccs + m];
			crate::L2::sscale_y(beta, seg);
			for l in 0..k {
				let col = if ta { &wa[l * mt..l * mt + mt] } else { &a[l * acs + m_mr..l * acs + m] };
				crate::L1::saxpy_k(alpha * bo(l, j), col, seg);
			}
		}
	}
	// column tail
	for jj in 0..nt {
		let j = n4 + jj;
		if ta {
			// rows below m_mr came through the padded group (or there were
			// none); the row tail is the same per-column axpy sequence
			let seg = &mut c[j * ccs + m_mr..j * ccs + m];
			crate::L2::sscale_y(beta, seg);
			for l in 0..k {
				crate::L1::saxpy_k(alpha * bo(l, j), &wa[l * mt..l * mt + mt], seg);
			}
		} else {
			let bcol: &[f32] = if tb { &wb[jj * k..jj * k + k] } else { &b[j * bcs..j * bcs + k] };
			sgemv_k(alpha, m, k, a, acs, bcol, beta, &mut c[j * ccs..j * ccs + m]);
		}
	}
}

/// The workspace layout of [`sgemm_packed_op`]: B panel, A panel,
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
		if main { KC * bpn } else { 0 },
		if main { MC * KC } else { 0 },
		if ta { mt * k } else { 0 },
		if tb && !pad { k * nt } else { 0 },
		if pad { 4 * m_mr } else { 0 },
	]
}

/// Elements of scratch [`sgemm_packed_op`] needs for these flags and
/// dimensions; equal to [`sgemm_packed_work`] when both flags are
/// `NoTrans`.
pub fn sgemm_packed_op_work(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> usize {
	packed_op_layout(transa, transb, m, n, k).iter().sum()
}
