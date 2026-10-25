//! `dgemm` — matrix multiplication: C ← αAB + βC.
//!
//! Implementation: the BLIS-style packed-panel shape with the K2
//! micro-kernel at every size (`dgemm_packed_bcast`: α folded into the
//! B panel at pack time, B broadcast by lane shuffles, two loads of A,
//! two of B, four shuffles, eight multiplies and eight adds per k step);
//! transposed operands are packed from a transposed read
//! (`dgemm_packed_op`). Every shape is bit-identical to the plain
//! dgemv-per-column reference, kept as `dgemm_colaxpy`; the losers of
//! the races stay exported: `dgemm_tiled` (the 4×4 register tile that
//! served below 1.5 MB of A until K2 beat it 1.2–2.3× at every size),
//! `dgemm_col4` (the fused stream) and `dgemm_packed` (the packed shape
//! with the pre-K2 kernel, 1.8–1.9× slower on the runner, two draws:
//! `bench/README.md`).

use super::super::check_mat;
use crate::{Error, Trans};
use crate::kernels::daxpy4;
use crate::L2::dgemv_k;

// Packed-panel sizes and the routing threshold, shared by the
// dispatcher, the packed shape and the workspace query.
pub(crate) const KC: usize = 256;
pub(crate) const MC: usize = 128;
/// The K2 candidate's block sizes: swept over KC 120–256 and MC 64–256
/// in the dev container with no measurable difference, so they stay at
/// the shipping values; never above KC/MC so [`dgemm_packed_work`]
/// covers both shapes.
const KC2: usize = KC;
const MC2: usize = MC;
pub(crate) const MR: usize = 4;

/// `C ← α·op(A)·op(B) + βC` with `op(A)` `m×k` and `op(B)` `k×n`, `C`
/// `m×n`, each at its own leading dimension. `op(X)` is `X`, `Xᵀ` or
/// `Xᴴ` per `transa`/`transb`. `work` is scratch of at least
/// [`dgemm_work`]`(transa, transb, m, n, k)` elements; it may be empty
/// when the query says zero.
///
/// A transposed operand is packed straight from a transposed read
/// ([`dgemm_packed_op`], decision D3's winner), so every flag
/// combination equals the unflagged call on an explicit copy, bit for
/// bit; `ConjTrans` is `Trans` for a real matrix. The copy candidate
/// ([`dgemm_copy_op`]) stays exported as the race's loser.
///
/// Errors: a leading dimension below its stored row count (arguments 8,
/// 10, 13); `a`, `b` or `c` shorter than its extent (arguments 7, 9, 12);
/// `work` shorter than the query (argument 14).
#[allow(clippy::too_many_arguments)]
pub fn dgemm(transa: Trans, transb: Trans, m: usize, n: usize, k: usize, alpha: f64, a: &[f64], lda: usize, b: &[f64], ldb: usize, beta: f64, c: &mut [f64], ldc: usize, work: &mut [f64]) -> Result<(), Error> {
	const R: &str = "dgemm";
	let (ar, ac) = if transa == Trans::NoTrans { (m, k) } else { (k, m) };
	let (br, bc) = if transb == Trans::NoTrans { (k, n) } else { (n, k) };
	crate::error::mat(R, 7, 8, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 9, 10, b.len(), br, bc, ldb)?;
	crate::error::mat(R, 12, 13, c.len(), m, n, ldc)?;
	crate::error::work(R, 14, work.len(), dgemm_work(transa, transb, m, n, k))?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	if k == 0 || alpha == 0.0 {
		// reference: C ← βC without reading A or B (β = 1 leaves C as
		// is, β = 0 writes zeros), the same per-column scaling every
		// shape applies before accumulating
		for j in 0..n {
			crate::L2::dscale_y(beta, &mut c[j * ldc..j * ldc + m]);
		}
		return Ok(());
	}
	// A transposed operand is packed straight from a transposed read
	// (`dgemm_packed_op`, the race's winner: `bench/README.md`); the
	// copy candidate (`dgemm_copy_op`) is the fallback `packed_route`
	// no longer selects. Both are bit-identical to the unflagged call on
	// an explicit copy (tested); conjugation is an exact sign flip in
	// either.
	if (transa != Trans::NoTrans || transb != Trans::NoTrans) && packed_route(m, k) {
		dgemm_packed_op(transa, transb, alpha, m, n, k, a, lda, b, ldb, beta, c, ldc, work);
		return Ok(());
	}
	dgemm_copy_op(transa, transb, alpha, m, n, k, a, lda, b, ldb, beta, c, ldc, work);
	Ok(())
}

/// Elements of scratch [`dgemm`] needs for these flags and dimensions:
/// [`dgemm_packed_work`] (unflagged) or [`dgemm_packed_op_work`] (a
/// transposed operand), the panels sized to the problem; zero when a
/// dimension is zero.
pub fn dgemm_work(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> usize {
	if m == 0 || n == 0 || k == 0 {
		return 0; // the quick returns compute nothing
	}
	if packed_route(m, k) && (transa != Trans::NoTrans || transb != Trans::NoTrans) {
		dgemm_packed_op_work(transa, transb, m, n, k)
	} else {
		dgemm_copy_op_work(transa, transb, m, n, k)
	}
}

/// Elements of scratch [`dgemm_packed`] needs for these dimensions:
/// the B panel (`KC` rows of the `n − n % 4` packed columns) and the
/// A panel (`MC × KC`); zero when the packed loops have nothing to do.
pub fn dgemm_packed_work(m: usize, n: usize, k: usize) -> usize {
	let n4 = n - n % 4;
	if k == 0 || n4 == 0 || m < MR {
		0
	} else {
		let kc = KC.min(k);
		kc * n4 + MC.min(m - m % MR) * kc
	}
}

/// Whether the dispatcher routes `m×k` `A` to the packed-panel shape.
#[inline]
pub(crate) fn packed_route(_m: usize, _k: usize) -> bool {
	// since the K2 race the packed shape wins at every size (1.2× over the
	// register tile at 128³ and widening above), so the tile's zone is gone;
	// the predicate stays for the transposed forms' routing and the queries
	true
}

/// C ← αAB + βC. A is m×k, B is k×n, C is m×n; each matrix has its own
/// column stride.
///
/// Runs the packed-panel shape with the K2 broadcast kernel
/// ([`dgemm_packed_bcast`]) at every size. Every exported shape is
/// bit-for-bit identical to it, so the routing is invisible to results.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dgemm_k(
	alpha: f64,
	m: usize,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	work: &mut [f64],
) {
	// The races that settled this, oldest first: the packed shape replaced
	// col4 above A ≈ 1.5 MB (two runner draws, 1.24× at 512³, 1.44× at
	// 1024³); the K2 kernel then beat the old packed shape 1.8–1.9× (two
	// runner draws) and the register tile 1.2–2.3× from 128³ up (dev
	// container), closing the tile's zone. `bench/README.md` has the tables.
	dgemm_packed_bcast(alpha, m, n, k, a, acs, b, bcs, beta, c, ccs, work);
}

/// The original column-daxpy shape (dgemv per column) — kept as the
/// plain reference the tuned shapes are raced and bit-checked against.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_colaxpy(
	alpha: f64,
	m: usize,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	for j in 0..n {
		dgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// Tuning-campaign candidate: 4×4 register-tiled dgemm.
/// The column-daxpy dgemm re-reads and re-writes each C element once per
/// k step; this micro-kernel holds a 4-row × 4-column tile of C in 8
/// SIMD registers across the whole k loop — one C load and one C store
/// per tile, 16 FLOPs per 2 A-register loads. Rounding sequence per
/// element is IDENTICAL to the column-daxpy path (β first, then one
/// α·b rounding and one multiply-add per k, ascending), so the two are
/// bit-for-bit interchangeable — tested. Tails (m%4, n%4) fall back to
/// the dgemv path.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_tiled(
	alpha: f64,
	m: usize,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		let mut i = 0usize;
		while i + 4 <= m {
			unsafe {
				tile_4x4(
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
			i += 4;
		}
		// row tail for these four columns: same per-element sequence as
		// the dgemv path over the remaining rows
		if i < m {
			for jj in j..j + 4 {
				let seg = &mut c[jj * ccs + i..jj * ccs + m];
				crate::L2::dscale_y(beta, seg);
				for l in 0..k {
					crate::L1::daxpy_k(
						alpha * b[jj * bcs + l],
						&a[l * acs + i..l * acs + m],
						seg,
					);
				}
			}
		}
		j += 4;
	}
	// column tail: plain dgemv columns
	while j < n {
		dgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate 2: 4-column fused dgemm.
/// The 4×4 tile's k-loop walks A at column stride (TLB/prefetch-hostile
/// at large n — measured losing above n≈512); this shape instead
/// streams each A column SEQUENTIALLY, once per group of four C
/// columns — A traffic drops 4× vs column-daxpy while the four hot C
/// columns ride the near caches. Rounding sequence per element is
/// identical to the column-daxpy path — bit-for-bit interchangeable,
/// tested.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_col4(
	alpha: f64,
	m: usize,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
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
				crate::L2::dscale_y(beta, col);
			}
			for l in 0..k {
				let t = [
					alpha * b[j * bcs + l],
					alpha * b[(j + 1) * bcs + l],
					alpha * b[(j + 2) * bcs + l],
					alpha * b[(j + 3) * bcs + l],
				];
				unsafe {
					daxpy4(
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
		dgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
		j += 1;
	}
}

/// Tuning-campaign candidate 3: packed-panel dgemm —
/// BLIS/Goto structure around the same 4×4 microkernel math. The tile's
/// weakness is its k-walk over A at column stride (TLB/prefetch-hostile
/// once A outgrows the near caches — why the dispatch abandons it at
/// 1.5 MB); this shape first copies a KC-deep block of B and an MC×KC
/// block of A into contiguous packed panels, so the microkernel streams
/// both operands sequentially regardless of matrix size. Motivating
/// evidence: deep-K prefill shapes (k ≥ 2048), where a packed-panel
/// competitor beat the col4 stream ~2× in a container probe; the
/// runner race (two draws, `bench/README.md`) confirmed it and this
/// shape is routed above the threshold.
///
/// Rounding sequence per element is IDENTICAL to the column-daxpy path:
/// k-blocking resumes each element's accumulation from its stored
/// partial, so the additions happen in the same ascending-k order with
/// the same once-rounded α·b terms, β applied exactly once (first
/// k-block) — bit-for-bit interchangeable, tested. Tails (m%4 rows,
/// n%4 columns) fall back to the same per-column sequences the tile
/// path uses.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_packed(
	alpha: f64,
	m: usize,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	work: &mut [f64],
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let need = dgemm_packed_work(m, n, k);
	assert!(work.len() >= need, "dgemm_packed: workspace shorter than dgemm_packed_work");
	// Panel sizes: Ap = MC·KC·8 = 256 KB (an L2-resident block), one
	// Bp column-group strip = KC·4·8 = 8 KB (L1-resident). First-guess
	// parameters — the race measures the shape, a later sweep can tune
	// the numbers.
	if k == 0 {
		// degenerate case the packed loops never reach: C ← βC
		for j in 0..n {
			crate::L2::dscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	if n4 > 0 && m_mr > 0 {
		// the caller's workspace, split into the two panels; every
		// element the microkernel reads is packed first, so the
		// slice's prior contents never reach C (tested with a NaN fill)
		let (bp, ap) = work[..need].split_at_mut(KC.min(k) * n4);
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			// pack B[pc.., 0..n4]: per 4-column group, k-major ×4
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
				// pack A[ic..ic+mc, pc..pc+kc]: per MR-row group, k-major ×MR
				for g in 0..mc / MR {
					let base = g * kc * MR;
					for l in 0..kc {
						let src = (pc + l) * acs + ic + g * MR;
						ap[base + l * MR..base + l * MR + MR]
							.copy_from_slice(&a[src..src + MR]);
					}
				}
				// jg outer / row-group inner: Ap rides L2 across column
				// groups, each 8 KB Bp strip rides L1 across row groups
				for jg in 0..n4 / 4 {
					for g in 0..mc / MR {
						unsafe {
							tile_4x4_packed(
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
	// row tail (m_mr..m) × packed columns: per-column daxpy over full k
	if m_mr < m {
		for j in 0..n4 {
			let seg = &mut c[j * ccs + m_mr..j * ccs + m];
			crate::L2::dscale_y(beta, seg);
			for l in 0..k {
				crate::L1::daxpy_k(alpha * b[j * bcs + l], &a[l * acs + m_mr..l * acs + m], seg);
			}
		}
	}
	// column tail: plain dgemv columns over all rows
	for j in n4..n {
		dgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// The K2 candidate: [`dgemm_packed`]'s blocking and panels with
/// OpenBLAS's wasm micro-kernel mechanics — α is folded into the B
/// panel once at pack time, and the k loop loads the four B values of
/// a column group as two vectors and broadcasts each lane with a
/// shuffle, so the loop holds no scalar loads, scalar multiplies or
/// scalar-to-vector moves. Per-element rounding sequence identical to
/// [`dgemm_packed`] (βC, then ascending k adds of (α·b)·a): bit-for-bit
/// interchangeable, tested. Raced against [`dgemm_packed`] and against
/// OpenBLAS compiled to wasm (`bench/README.md`).
#[allow(clippy::too_many_arguments)]
pub fn dgemm_packed_bcast(
	alpha: f64,
	m: usize,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	work: &mut [f64],
) {
	check_mat(a.len(), m, k, acs);
	check_mat(b.len(), k, n, bcs);
	check_mat(c.len(), m, n, ccs);
	let need = dgemm_packed_work(m, n, k);
	assert!(work.len() >= need, "dgemm_packed_bcast: workspace shorter than dgemm_packed_work");
	// Panel sizes: Ap = MC·KC·8 = 256 KB (an L2-resident block), one
	// Bp column-group strip = KC·4·8 = 8 KB (L1-resident). First-guess
	// parameters — the race measures the shape, a later sweep can tune
	// the numbers.
	if k == 0 {
		// degenerate case the packed loops never reach: C ← βC
		for j in 0..n {
			crate::L2::dscale_y(beta, &mut c[j * ccs..j * ccs + m]);
		}
		return;
	}
	let n4 = n - n % 4;
	let m_mr = m - m % MR;
	if n4 > 0 && m_mr > 0 {
		// the caller's workspace, split into the two panels; every
		// element the microkernel reads is packed first, so the
		// slice's prior contents never reach C (tested with a NaN fill)
		let (bp, ap) = work[..need].split_at_mut(KC2.min(k) * n4);
		let mut pc = 0usize;
		while pc < k {
			let kc = KC2.min(k - pc);
			// pack B[pc.., 0..n4]: per 4-column group, k-major ×4
			for jg in 0..n4 / 4 {
				let dst = &mut bp[jg * kc * 4..(jg + 1) * kc * 4];
				let j = jg * 4;
				for l in 0..kc {
					for u in 0..4 {
						dst[l * 4 + u] = alpha * b[(j + u) * bcs + pc + l]; // α folded in once
					}
				}
			}
			let first = pc == 0;
			let mut ic = 0usize;
			while ic < m_mr {
				let mc = MC2.min(m_mr - ic); // both multiples of MR
				// pack A[ic..ic+mc, pc..pc+kc]: per MR-row group, k-major ×MR
				for g in 0..mc / MR {
					let base = g * kc * MR;
					for l in 0..kc {
						let src = (pc + l) * acs + ic + g * MR;
						ap[base + l * MR..base + l * MR + MR]
							.copy_from_slice(&a[src..src + MR]);
					}
				}
				// jg outer / row-group inner: Ap rides L2 across column
				// groups, each 8 KB Bp strip rides L1 across row groups
				for jg in 0..n4 / 4 {
					for g in 0..mc / MR {
						unsafe {
							tile_4x4_packed_bcast(
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
	// row tail (m_mr..m) × packed columns: per-column daxpy over full k
	if m_mr < m {
		for j in 0..n4 {
			let seg = &mut c[j * ccs + m_mr..j * ccs + m];
			crate::L2::dscale_y(beta, seg);
			for l in 0..k {
				crate::L1::daxpy_k(alpha * b[j * bcs + l], &a[l * acs + m_mr..l * acs + m], seg);
			}
		}
	}
	// column tail: plain dgemv columns over all rows
	for j in n4..n {
		dgemv_k(alpha, m, k, a, acs, &b[j * bcs..j * bcs + k], beta, &mut c[j * ccs..j * ccs + m]);
	}
}

/// The 4×4 microkernel of [`dgemm_packed_bcast`]: `bp` holds α·B already;
/// each k step is two loads of A, two loads of B, four lane broadcasts,
/// eight multiplies and eight adds.
/// # Safety
/// As [`tile_4x4_packed`].
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn tile_4x4_packed_bcast(
	kc: usize,
	ap: *const f64,
	bp: *const f64,
	beta: f64,
	first: bool,
	cp: *mut f64,
	ccs: usize,
) {
	use crate::lanes::F64x2;
	let zero = F64x2::splat(0.0);
	let mut acc = [[zero; 2]; 4];
	if first {
		if beta != 0.0 {
			let vb = F64x2::splat(beta);
			for (u, au) in acc.iter_mut().enumerate() {
				au[0] = F64x2::load(cp.add(u * ccs)).mul(vb);
				au[1] = F64x2::load(cp.add(u * ccs + 2)).mul(vb);
			}
		}
	} else {
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = F64x2::load(cp.add(u * ccs));
			au[1] = F64x2::load(cp.add(u * ccs + 2));
		}
	}
	for l in 0..kc {
		let a0 = F64x2::load(ap.add(l * 4));
		let a1 = F64x2::load(ap.add(l * 4 + 2));
		let b01 = F64x2::load(bp.add(l * 4));
		let b23 = F64x2::load(bp.add(l * 4 + 2));
		let t = [b01.dup0(), b01.dup1(), b23.dup0(), b23.dup1()];
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = au[0].add(a0.mul(t[u]));
			au[1] = au[1].add(a1.mul(t[u]));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cp.add(u * ccs));
		au[1].store(cp.add(u * ccs + 2));
	}
}


/// One 4×4 microkernel step over PACKED panels: `ap` walks MR=4
/// contiguous row values per k step, `bp` walks 4 contiguous column
/// values per k step. Math and per-element rounding order are identical
/// to [`tile_4x4`]; `first` selects β-application (first k-block) vs
/// continuing accumulation from the stored partial.
/// # Safety
/// `ap` must be valid for `kc*4` f64s, `bp` for `kc*4`; `cp` points at
/// `C[i, j]` (stride ccs) with 4 rows and 4 columns in bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_4x4_packed(
	alpha: f64,
	kc: usize,
	ap: *const f64,
	bp: *const f64,
	beta: f64,
	first: bool,
	cp: *mut f64,
	ccs: usize,
) {
	use crate::lanes::F64x2;
	let zero = F64x2::splat(0.0);
	let mut acc = [[zero; 2]; 4];
	if first {
		if beta != 0.0 {
			let vb = F64x2::splat(beta);
			for (u, au) in acc.iter_mut().enumerate() {
				au[0] = F64x2::load(cp.add(u * ccs)).mul(vb);
				au[1] = F64x2::load(cp.add(u * ccs + 2)).mul(vb);
			}
		}
	} else {
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = F64x2::load(cp.add(u * ccs));
			au[1] = F64x2::load(cp.add(u * ccs + 2));
		}
	}
	for l in 0..kc {
		let a0 = F64x2::load(ap.add(l * 4));
		let a1 = F64x2::load(ap.add(l * 4 + 2));
		for (u, au) in acc.iter_mut().enumerate() {
			let t = F64x2::splat(alpha * *bp.add(l * 4 + u));
			au[0] = au[0].add(a0.mul(t));
			au[1] = au[1].add(a1.mul(t));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cp.add(u * ccs));
		au[1].store(cp.add(u * ccs + 2));
	}
}

/// One 4×4 tile: rows i..i+4 (two f64x2 registers) × columns j..j+4.
/// # Safety
/// `ap` points at `A[i, 0]` (stride acs, k columns), `bp` at `B[0, j]`
/// (stride bcs), `cp` at `C[i, j]` (stride ccs); 4 rows and 4 B/C
/// columns must be in bounds.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_4x4(
	alpha: f64,
	k: usize,
	ap: *const f64,
	acs: usize,
	bp: *const f64,
	bcs: usize,
	beta: f64,
	cp: *mut f64,
	ccs: usize,
) {
	use crate::lanes::F64x2;
	let zero = F64x2::splat(0.0);
	let vb = F64x2::splat(beta);
	let mut acc = [[zero; 2]; 4];
	if beta != 0.0 {
		for (u, au) in acc.iter_mut().enumerate() {
			au[0] = F64x2::load(cp.add(u * ccs)).mul(vb);
			au[1] = F64x2::load(cp.add(u * ccs + 2)).mul(vb);
		}
	}
	for l in 0..k {
		let a0 = F64x2::load(ap.add(l * acs));
		let a1 = F64x2::load(ap.add(l * acs + 2));
		for (u, au) in acc.iter_mut().enumerate() {
			let t = F64x2::splat(alpha * *bp.add(u * bcs + l));
			au[0] = au[0].add(a0.mul(t));
			au[1] = au[1].add(a1.mul(t));
		}
	}
	for (u, au) in acc.iter().enumerate() {
		au[0].store(cp.add(u * ccs));
		au[1].store(cp.add(u * ccs + 2));
	}
}

/// The other candidate for the transposed forms (decision D3 in
/// `DESIGN.md`; the race in `bench/README.md`): a transposed operand is transposed into `work`
/// with `transpose_into` and the product then runs the no-transpose
/// shape on the copy, so every flag combination is the explicit-copy
/// definition itself. The netlib wrapper routes it only for the
/// no-transpose call (where it coincides with the plain shape); a
/// transposed operand goes to [`dgemm_packed_op`], the race's choice.
/// `work` needs [`dgemm_copy_op_work`]
/// elements. Exported so the benchmarks can race it; it asserts rather
/// than returning errors.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_copy_op(
	transa: Trans,
	transb: Trans,
	alpha: f64,
	m: usize,
	n: usize,
	k: usize,
	a: &[f64],
	lda: usize,
	b: &[f64],
	ldb: usize,
	beta: f64,
	c: &mut [f64],
	ldc: usize,
	work: &mut [f64],
) {
	assert!(work.len() >= dgemm_copy_op_work(transa, transb, m, n, k), "dgemm_copy_op: workspace shorter than dgemm_copy_op_work");
	if m == 0 || n == 0 || k == 0 {
		// nothing to copy: C ← βC as every shape does before accumulating
		for j in 0..n {
			crate::L2::dscale_y(beta, &mut c[j * ldc..j * ldc + m]);
		}
		return;
	}
	let (work, a, lda) = if transa == Trans::NoTrans {
		(work, a, lda)
	} else {
		let (ta, rest) = work.split_at_mut(m * k);
		super::super::transpose_into(a, lda, k, m, ta, |x| x);
		(rest, &*ta, m)
	};
	let (work, b, ldb) = if transb == Trans::NoTrans {
		(work, b, ldb)
	} else {
		let (tb, rest) = work.split_at_mut(k * n);
		super::super::transpose_into(b, ldb, n, k, tb, |x| x);
		(rest, &*tb, k)
	};
	dgemm_k(alpha, m, n, k, a, lda, b, ldb, beta, c, ldc, work);
}

/// Elements of scratch [`dgemm_copy_op`] needs: the copies (`m·k` for a
/// transposed `A`, `k·n` for a transposed `B`) plus [`dgemm_packed_work`]
/// when the dispatcher routes to the packed shape.
pub fn dgemm_copy_op_work(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> usize {
	if m == 0 || n == 0 || k == 0 {
		return 0;
	}
	let ta = if transa == Trans::NoTrans { 0 } else { m * k };
	let tb = if transb == Trans::NoTrans { 0 } else { k * n };
	let packed = if packed_route(m, k) { dgemm_packed_work(m, n, k) } else { 0 };
	ta + tb + packed
}

/// The transposed forms (the race's winner over a whole-operand copy,
/// two runner draws: `bench/README.md`): the
/// packed-panel shape with `op(A)` and
/// `op(B)` packed straight from a transposed read — op(A)'s row is a
/// stored column of A and op(B)'s packed quad is four consecutive stored
/// elements — so no whole-operand copy is made and the microkernel is
/// unchanged. The three tails that read the operands directly are kept
/// bit-identical by small gathers into the workspace: op(A)'s row tail
/// (`m % MR` rows × k), op(B)'s column tail (k × `n % 4`), and, for a
/// transposed A, the column tail's first `m − m % MR` rows ride a fourth
/// packed column group padded with zeros into a `4 × (m − m % MR)`
/// scratch block of C. `work` needs [`dgemm_packed_op_work`] elements.
/// Bit-identical to [`dgemm_packed`] on an explicit transpose (tested);
/// with both flags `NoTrans` it is [`dgemm_packed`] exactly.
#[allow(clippy::too_many_arguments)]
pub fn dgemm_packed_op(
	transa: Trans,
	transb: Trans,
	alpha: f64,
	m: usize,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	b: &[f64],
	bcs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	work: &mut [f64],
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
	assert!(work.len() >= need, "dgemm_packed_op: workspace shorter than dgemm_packed_op_work");
	if k == 0 {
		for j in 0..n {
			crate::L2::dscale_y(beta, &mut c[j * ccs..j * ccs + m]);
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
						for u in 0..4 {
							dst[l * 4 + u] = alpha * b[(pc + l) * bcs + j + u];
						}
					}
				} else {
					for l in 0..kc {
						for u in 0..4 {
							dst[l * 4 + u] = alpha * b[(j + u) * bcs + pc + l];
						}
					}
				}
			}
			if pad {
				let dst = &mut bp[(n4 / 4) * kc * 4..(n4 / 4 + 1) * kc * 4];
				for l in 0..kc {
					for u in 0..4 {
						dst[l * 4 + u] = if u < nt { alpha * bo(pc + l, n4 + u) } else { 0.0 };
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
							tile_4x4_packed_bcast(
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
							tile_4x4_packed_bcast(
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
			crate::L2::dscale_y(beta, seg);
			for l in 0..k {
				let col = if ta { &wa[l * mt..l * mt + mt] } else { &a[l * acs + m_mr..l * acs + m] };
				crate::L1::daxpy_k(alpha * bo(l, j), col, seg);
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
			crate::L2::dscale_y(beta, seg);
			for l in 0..k {
				crate::L1::daxpy_k(alpha * bo(l, j), &wa[l * mt..l * mt + mt], seg);
			}
		} else {
			let bcol: &[f64] = if tb { &wb[jj * k..jj * k + k] } else { &b[j * bcs..j * bcs + k] };
			dgemv_k(alpha, m, k, a, acs, bcol, beta, &mut c[j * ccs..j * ccs + m]);
		}
	}
}

/// The workspace layout of [`dgemm_packed_op`]: B panel, A panel,
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
	let kc = KC.min(k);
	[
		if main { kc * bpn } else { 0 },
		if main { MC.min(m_mr) * kc } else { 0 },
		if ta { mt * k } else { 0 },
		if tb && !pad { k * nt } else { 0 },
		if pad { 4 * m_mr } else { 0 },
	]
}

/// Elements of scratch [`dgemm_packed_op`] needs for these flags and
/// dimensions; equal to [`dgemm_packed_work`] when both flags are
/// `NoTrans`.
pub fn dgemm_packed_op_work(transa: Trans, transb: Trans, m: usize, n: usize, k: usize) -> usize {
	packed_op_layout(transa, transb, m, n, k).iter().sum()
}
