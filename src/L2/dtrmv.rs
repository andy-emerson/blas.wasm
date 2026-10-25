//! `dtrmv` — triangular matrix × vector, in place: x ← Ax.
//!
//! Implementation: 4-column fan-in column-daxpy —
//! groups of four columns share one pass over x's common segment
//! (`kernels::daxpy4in`, sources in reference `dtrmv`'s column order:
//! ascending for upper, descending for lower), so x's
//! read-modify-write traffic drops 4×; the ≤3-row ragged band and the
//! diagonal writes run scalar in the original step order. Each `x[j]`
//! is still consumed before its own slot is overwritten (the four
//! group t-values are provably untouched until their step), and the
//! per-element rounding sequence is unchanged — bit-for-bit the plain
//! column loop, locked by the existing test. The transposed
//! form is a dot-fold reduction per stored column (`dtrmv_t_k`).

use super::check_mat;
use crate::L1::ddot_k;
use crate::{Diag, Error, Trans, Uplo};
use crate::kernels::daxpy4in;
use crate::L1::daxpy_k;

/// `x ← op(A)·x` for a triangular `A` of order `n`, the triangle named by
/// `uplo` stored at leading dimension `lda`, with a unit diagonal when
/// `diag` says so; `x` has `n` elements. `op(A)` is `A`, `Aᵀ` or `Aᴴ`
/// per `trans`.
///
/// The transposed forms (`Trans::Trans`; `ConjTrans` is the same for a real
/// matrix) take each new element as the `ddot`
/// fold of a stored column's triangle segment with `x`: a different,
/// documented rounding order from the no-transpose sweep, replayed bit
/// for bit by the tests.
///
/// Errors: `lda < max(1, n)` (argument 6); `a` or `x` shorter than its
/// extent (arguments 5, 7).
pub fn dtrmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: &[f64], lda: usize, x: &mut [f64]) -> Result<(), Error> {
	const R: &str = "dtrmv";
	crate::error::mat(R, 5, 6, a.len(), n, n, lda)?;
	crate::error::vec(R, 7, x.len(), n)?;
	if n == 0 {
		return Ok(());
	}
	match trans {
		Trans::NoTrans => dtrmv_k(n, a, lda, uplo.is_upper(), diag.is_unit(), &mut x[..n]),
		_ => dtrmv_t_k(n, a, lda, uplo.is_upper(), diag.is_unit(), &mut x[..n]),
	}
	Ok(())
}

/// x ← Ax, A triangular n×n at column stride `cs`. `upper` selects the
/// triangle; `unit` treats the diagonal as ones (stored values
/// ignored).
pub(crate) fn dtrmv_k(n: usize, a: &[f64], cs: usize, upper: bool, unit: bool, x: &mut [f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "dtrmv: x length mismatch");
	let ap = a.as_ptr();
	if upper {
		let mut j = 0usize;
		while j + 4 <= n {
			// all four t's are original: step j+u writes only x[..=j+u']
			// for u' < u, never x[j+u]
			let t = [x[j], x[j + 1], x[j + 2], x[j + 3]];
			unsafe {
				daxpy4in(
					ap.add(j * cs),
					ap.add((j + 1) * cs),
					ap.add((j + 2) * cs),
					ap.add((j + 3) * cs),
					t,
					x.as_mut_ptr(),
					j,
				);
			}
			// ragged band + diagonals, original ascending step order
			for u in 0..4 {
				let cj = (j + u) * cs;
				for i in j..j + u {
					x[i] += t[u] * a[cj + i];
				}
				if !unit {
					x[j + u] = t[u] * a[cj + j + u];
				}
			}
			j += 4;
		}
		while j < n {
			let col = &a[j * cs..j * cs + n];
			let t = x[j];
			daxpy_k(t, &col[..j], &mut x[..j]);
			if !unit {
				x[j] = t * col[j];
			}
			j += 1;
		}
	} else {
		let r = n % 4;
		// high leftover columns first, original descending order
		for j in (n - r..n).rev() {
			let col = &a[j * cs..j * cs + n];
			let t = x[j];
			daxpy_k(t, &col[j + 1..], &mut x[j + 1..]);
			if !unit {
				x[j] = t * col[j];
			}
		}
		let mut g = n - r;
		while g >= 4 {
			g -= 4;
			// steps g+3 down to g; all four t's are original (step g+u'
			// writes only x[g+u'+1..], never x[g..=g+u'])
			let t = [x[g], x[g + 1], x[g + 2], x[g + 3]];
			// common suffix x[g+4..], sources in descending step order
			unsafe {
				daxpy4in(
					ap.add((g + 3) * cs + g + 4),
					ap.add((g + 2) * cs + g + 4),
					ap.add((g + 1) * cs + g + 4),
					ap.add(g * cs + g + 4),
					[t[3], t[2], t[1], t[0]],
					x.as_mut_ptr().add(g + 4),
					n - g - 4,
				);
			}
			// ragged band + diagonals, original descending step order
			for u in (0..4).rev() {
				let cj = (g + u) * cs;
				for i in g + u + 1..g + 4 {
					x[i] += t[u] * a[cj + i];
				}
				if !unit {
					x[g + u] = t[u] * a[cj + g + u];
				}
			}
		}
	}
}

/// `x ← op(A)·x` with `op(A) = Aᵀ`, A triangular
/// n×n at column stride `cs`; `upper` names the STORED triangle (op(A)
/// holds the other). Per element: `x[j] ← d·x[j] + t` (`d` the diagonal, or the bare `x[j]` for a unit
/// one) with `t` the fold of stored column j's triangle segment against
/// the original `x`, the fold being the `ddot`
/// kernel's register-parallel order (`src/L1`), so the contract is a
/// deterministic reduction order rather than the sweep's sequence — the
/// tests replay that fold bit for bit. No singularity check.
pub(crate) fn dtrmv_t_k(n: usize, a: &[f64], cs: usize, upper: bool, unit: bool, x: &mut [f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "dtrmv_t: x length mismatch");
	if upper {
		// stored upper is op(A) lower: x[j] depends on the original x[..j],
		// so columns go descending and every read is still original
		for j in (0..n).rev() {
			let col = &a[j * cs..j * cs + j];
			let xs = &x[..j];
			let t = ddot_k(col, xs);
			let d = if unit { x[j] } else { x[j] * a[j * cs + j] };
			x[j] = d + t;
		}
	} else {
		// stored lower is op(A) upper: x[j] depends on the original
		// x[j + 1..], so columns go ascending
		for j in 0..n {
			let col = &a[j * cs + j + 1..j * cs + n];
			let xs = &x[j + 1..n];
			let t = ddot_k(col, xs);
			let d = if unit { x[j] } else { x[j] * a[j * cs + j] };
			x[j] = d + t;
		}
	}
}
