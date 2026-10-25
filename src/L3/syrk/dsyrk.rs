//! `dsyrk` — symmetric rank-k (Gram-matrix) update:
//! C ← αAAᵀ + βC, one triangle of C stored.
//!
//! Implementation: the packed gemm tile over the stored triangle
//! (`dsyrk_packed`: A's rows as both panels, α folded into the right one,
//! the diagonal tiles through scratch); `dsyrk_sweep`, the fan-out sweep
//! that shipped before K1, stays exported as the tests' bit-identity
//! reference.
//!
//! The family's chapter — what the operation is, exactly which rounding
//! sequence every shape reproduces, how it is computed, what it calls and
//! how it performs — is this folder's README, the `syrk` module page.

use super::super::check_mat;
use super::super::dgemm::{tile_4x4_packed_bcast, KC, MC, MR};
use crate::{Error, Trans, Uplo};
use crate::kernels::daxpy4;
use crate::L1::daxpy_k;
use crate::L2::dscale_y;

/// `C ← α·op(A)·op(A)ᵀ + βC` with `op(A)` `n×k` for [`Trans::NoTrans`]
/// and `k×n` otherwise; `C` is symmetric of order `n` with the triangle
/// named by `uplo` stored and updated. `work` is scratch of at least
/// [`dsyrk_work`]`(trans, n, k)` elements.
///
/// The transposed form transposes `A` into the workspace first and
/// then runs the no-transpose kernel, so it equals the unflagged call on
/// an explicit copy bit for bit; `ConjTrans` is `Trans` for a real matrix.
///
/// Errors: a leading dimension below its row count (arguments 7, 10);
/// `a` or `c` shorter than its extent (arguments 6, 9); `work` shorter
/// than the query (argument 11).
#[allow(clippy::too_many_arguments)]
pub fn dsyrk(uplo: Uplo, trans: Trans, n: usize, k: usize, alpha: f64, a: &[f64], lda: usize, beta: f64, c: &mut [f64], ldc: usize, work: &mut [f64]) -> Result<(), Error> {
	const R: &str = "dsyrk";
	let (ar, ac) = if trans == Trans::NoTrans { (n, k) } else { (k, n) };
	crate::error::mat(R, 6, 7, a.len(), ar, ac, lda)?;
	crate::error::mat(R, 9, 10, c.len(), n, n, ldc)?;
	crate::error::work(R, 11, work.len(), dsyrk_work(trans, n, k))?;
	if n == 0 {
		return Ok(());
	}
	if alpha == 0.0 || k == 0 {
		// reference: the stored triangle of C ← βC without reading A
		for j in 0..n {
			let (lo, hi) = if uplo.is_upper() { (0, j + 1) } else { (j, n) };
			crate::L2::dscale_y(beta, &mut c[j * ldc + lo..j * ldc + hi]);
		}
		return Ok(());
	}
	// op(A) is materialized in the workspace ahead of the
	// no-transpose kernel, so the transposed form is bit-identical to
	// the unflagged call on an explicit copy (tested)
	if trans == Trans::NoTrans {
		dsyrk_packed(uplo.is_upper(), alpha, n, k, a, lda, beta, c, ldc, work);
	} else {
		let (ta, rest) = work.split_at_mut(n * k);
		super::super::transpose_into(a, lda, k, n, ta, |x| x);
		dsyrk_packed(uplo.is_upper(), alpha, n, k, ta, n, beta, c, ldc, rest);
	}
	Ok(())
}

/// Elements of scratch [`dsyrk`] needs for these flags and dimensions:
/// the packed panels ([`dsyrk_packed_work`]), plus `n·k` for the
/// transposed form (the copy of `op(A)`).
pub fn dsyrk_work(trans: Trans, n: usize, k: usize) -> usize {
	dsyrk_packed_work(n, k) + if trans == Trans::NoTrans { 0 } else { n * k }
}

/// C ← αAAᵀ + βC. A is n×k; C is n×n with the `upper` (or lower)
/// triangle stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dsyrk_k(
	alpha: f64,
	n: usize,
	k: usize,
	a: &[f64],
	acs: usize,
	beta: f64,
	c: &mut [f64],
	ccs: usize,
	upper: bool,
) {
	check_mat(a.len(), n, k, acs);
	check_mat(c.len(), n, n, ccs);
	let mut j = 0usize;
	while j + 4 <= n {
		let cp = c.as_mut_ptr();
		for u in 0..4 {
			let (lo, hi) = if upper { (0, j + u + 1) } else { (j + u, n) };
			let seg =
				unsafe { core::slice::from_raw_parts_mut(cp.add((j + u) * ccs + lo), hi - lo) };
			dscale_y(beta, seg);
		}
		// intersection of the four stored segments; the rest is the
		// ≤3-row ragged edge per column
		let (clo, chi) = if upper { (0, j + 1) } else { (j + 3, n) };
		for l in 0..k {
			let t = [
				alpha * a[l * acs + j],
				alpha * a[l * acs + j + 1],
				alpha * a[l * acs + j + 2],
				alpha * a[l * acs + j + 3],
			];
			unsafe {
				daxpy4(
					a.as_ptr().add(l * acs + clo),
					t,
					cp.add(j * ccs + clo),
					cp.add((j + 1) * ccs + clo),
					cp.add((j + 2) * ccs + clo),
					cp.add((j + 3) * ccs + clo),
					chi - clo,
				);
			}
			for (u, &tu) in t.iter().enumerate() {
				let (lo, hi) = if upper { (chi, j + u + 1) } else { (j + u, clo) };
				for i in lo..hi {
					unsafe {
						*cp.add((j + u) * ccs + i) += a[l * acs + i] * tu;
					}
				}
			}
		}
		j += 4;
	}
	while j < n {
		let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
		let seg = &mut c[j * ccs + lo..j * ccs + hi];
		dscale_y(beta, seg);
		for l in 0..k {
			daxpy_k(alpha * a[l * acs + j], &a[l * acs + lo..l * acs + hi], seg);
		}
		j += 1;
	}
}

/// The K1 candidate: `dsyrk` on the packing layer. `C ← βC + α·A·Aᵀ`
/// runs as the packed gemm with the B panel read from A's rows (op(B) =
/// Aᵀ, so `B[l, j] = A[j, l]`, α folded in at pack time) and
/// the broadcast tile of the packed gemm as the micro-kernel; a tile entirely in the
/// unstored triangle is skipped, a tile crossing the diagonal is
/// computed into a scratch tile and copied back only where stored, so
/// the unstored triangle is never written. The `n % MR` row tail and the
/// `n % 4` column tail run the plain column-daxpy sweeps. Per-element
/// rounding sequence identical to the shipping sweep (βC, then ascending-k adds
/// of `A[i,l]·(α·A[j,l])`): bit-for-bit interchangeable, tested. Raced
/// against the shipping sweep and against OpenBLAS compiled to wasm
/// (`bench/README.md`).
#[allow(clippy::too_many_arguments)]
pub fn dsyrk_packed(upper: bool, alpha: f64, n: usize, k: usize, a: &[f64], acs: usize, beta: f64, c: &mut [f64], ccs: usize, work: &mut [f64]) {
	check_mat(a.len(), n, k, acs);
	check_mat(c.len(), n, n, ccs);
	let need = dsyrk_packed_work(n, k);
	assert!(work.len() >= need, "dsyrk_packed: workspace shorter than dsyrk_packed_work");
	if k == 0 {
		for j in 0..n {
			let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
			dscale_y(beta, &mut c[j * ccs + lo..j * ccs + hi]);
		}
		return;
	}
	let n4 = n - n % 4;
	let n_mr = n - n % MR;
	// whether the MR×4 tile at (rows r0.., cols c0..) is all stored, all
	// unstored, or crosses the diagonal
	let tile_kind = |r0: usize, c0: usize| -> u8 {
		let (r1, c1) = (r0 + MR - 1, c0 + 3);
		if upper {
			if r1 <= c0 { 2 } else if r0 > c1 { 0 } else { 1 }
		} else if r0 >= c1 {
			2
		} else if r1 < c0 {
			0
		} else {
			1
		}
	};
	if n4 > 0 && n_mr > 0 {
		let kcm = KC.min(k);
		let (bp, rest) = work[..need].split_at_mut(kcm * n4);
		let (ap, dt) = rest.split_at_mut(MC.min(n_mr) * kcm);
		let mut pc = 0usize;
		while pc < k {
			let kc = KC.min(k - pc);
			// pack op(B) = Aᵀ: per 4-column group, k-major ×4, α folded in
			for jg in 0..n4 / 4 {
				let dst = &mut bp[jg * kc * 4..(jg + 1) * kc * 4];
				let j = jg * 4;
				for l in 0..kc {
					for u in 0..4 {
						dst[l * 4 + u] = alpha * a[(pc + l) * acs + j + u];
					}
				}
			}
			let first = pc == 0;
			let mut ic = 0usize;
			while ic < n_mr {
				let mc = MC.min(n_mr - ic);
				for g in 0..mc / MR {
					let base = g * kc * MR;
					for l in 0..kc {
						let src = (pc + l) * acs + ic + g * MR;
						ap[base + l * MR..base + l * MR + MR].copy_from_slice(&a[src..src + MR]);
					}
				}
				for jg in 0..n4 / 4 {
					for g in 0..mc / MR {
						let r0 = ic + g * MR;
						let c0 = jg * 4;
						match tile_kind(r0, c0) {
							0 => {}
							2 => unsafe {
								tile_4x4_packed_bcast(
									kc,
									ap.as_ptr().add(g * kc * MR),
									bp.as_ptr().add(jg * kc * 4),
									beta,
									first,
									c.as_mut_ptr().add(c0 * ccs + r0),
									ccs,
								);
							},
							_ => {
								// the diagonal tile: its scratch copy carries the
								// partials across k-blocks; the stored elements are
								// copied back after the last block
								let slot = (r0 / MR) * (MR / 4) + (c0 - r0) / 4; // the (MR/4) groups crossing this row block, in column order
								let st = &mut dt[slot * 4 * MR..(slot + 1) * 4 * MR];
								if first {
									for u in 0..4 {
										for r in 0..MR {
											let (i, j) = (r0 + r, c0 + u);
											let stored = if upper { i <= j } else { i >= j };
											st[u * MR + r] = if stored && beta != 0.0 { c[j * ccs + i] } else { 0.0 };
										}
									}
								}
								unsafe {
									tile_4x4_packed_bcast(
										kc,
										ap.as_ptr().add(g * kc * MR),
										bp.as_ptr().add(jg * kc * 4),
										beta,
										first,
										st.as_mut_ptr(),
										MR,
									);
								}
								if pc + kc >= k {
									for u in 0..4 {
										for r in 0..MR {
											let (i, j) = (r0 + r, c0 + u);
											let stored = if upper { i <= j } else { i >= j };
											if stored {
												c[j * ccs + i] = st[u * MR + r];
											}
										}
									}
								}
							}
						}
					}
				}
				ic += mc;
			}
			pc += kc;
		}
	}
	// row tail (n_mr..n) × packed columns, the stored rows of each
	// column: all of them below the diagonal, those down to the diagonal
	// above it (the 8-row tile leaves up to 7 tail rows, which can hold
	// stored upper elements of the packed columns)
	if n_mr < n {
		for j in 0..n4 {
			let (lo, hi) = if upper { (n_mr, (j + 1).max(n_mr)) } else { (n_mr.max(j), n) };
			if lo >= hi {
				continue;
			}
			let seg = &mut c[j * ccs + lo..j * ccs + hi];
			dscale_y(beta, seg);
			for l in 0..k {
				daxpy_k(alpha * a[l * acs + j], &a[l * acs + lo..l * acs + hi], seg);
			}
		}
	}
	// column tail: the plain sweep per stored column
	for j in n4..n {
		let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
		let seg = &mut c[j * ccs + lo..j * ccs + hi];
		dscale_y(beta, seg);
		for l in 0..k {
			daxpy_k(alpha * a[l * acs + j], &a[l * acs + lo..l * acs + hi], seg);
		}
	}
}

/// Elements of scratch [`dsyrk_packed`] needs: the B panel, the A panel
/// and one scratch tile per diagonal tile; zero when the packed loops
/// have nothing to do.
pub fn dsyrk_packed_work(n: usize, k: usize) -> usize {
	let n4 = n - n % 4;
	let n_mr = n - n % MR;
	if k == 0 || n4 == 0 || n_mr == 0 {
		0
	} else {
		let kc = KC.min(k);
		kc * n4 + MC.min(n_mr) * kc + MR * n_mr
	}
}

/// The shipping shape before K1, the truncated column-axpy sweep: kept
/// as the race's loser and the bit-identity reference of
/// [`dsyrk_packed`]; exported so the tests and the benchmarks can call
/// it. Asserts rather than returning errors.
#[allow(clippy::too_many_arguments)]
pub fn dsyrk_sweep(upper: bool, alpha: f64, n: usize, k: usize, a: &[f64], lda: usize, beta: f64, c: &mut [f64], ldc: usize) {
	dsyrk_k(alpha, n, k, a, lda, beta, c, ldc, upper);
}
