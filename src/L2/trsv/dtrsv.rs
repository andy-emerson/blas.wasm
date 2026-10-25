//! `dtrsv` — triangular solve, one right-hand side, in place: x ← A⁻¹x.
//!
//! Implementation: divide-then-column-daxpy, 4-column fan-in — the four divisions and the ≤3-row in-group
//! elimination band run scalar in reference `dtrsv`'s order (the
//! t-values are sequentially dependent), then the four solved
//! unknowns eliminate from the common remaining segment in one shared
//! pass (`kernels::daxpy4in`, sources in the same step order), cutting
//! x's read-modify-write traffic 4×. Per-element rounding sequence
//! unchanged — bit-for-bit the plain loop, locked by the existing
//! test. The transposed
//! form is a dot-fold reduction per stored column (`dtrsv_t_k`).

use super::super::check_mat;
use crate::L1::ddot_k;
use crate::{Diag, Error, Trans, Uplo};
use crate::kernels::daxpy4in;
use crate::L1::daxpy_k;

/// `x ← op(A)⁻¹·x` for a triangular `A` of order `n`, the triangle named by
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
pub fn dtrsv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: &[f64], lda: usize, x: &mut [f64]) -> Result<(), Error> {
	const R: &str = "dtrsv";
	crate::error::mat(R, 5, 6, a.len(), n, n, lda)?;
	crate::error::vec(R, 7, x.len(), n)?;
	if n == 0 {
		return Ok(());
	}
	match trans {
		Trans::NoTrans => dtrsv_k(n, a, lda, uplo.is_upper(), diag.is_unit(), &mut x[..n]),
		_ => dtrsv_t_k(n, a, lda, uplo.is_upper(), diag.is_unit(), &mut x[..n]),
	}
	Ok(())
}

/// x ← A⁻¹x, A triangular n×n at column stride `cs`. `upper` selects
/// the triangle; `unit` treats the diagonal as ones (stored values
/// ignored). No singularity check — a zero diagonal yields inf/NaN,
/// as in reference BLAS.
pub(crate) fn dtrsv_k(n: usize, a: &[f64], cs: usize, upper: bool, unit: bool, x: &mut [f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "dtrsv: x length mismatch");
	let ap = a.as_ptr();
	if upper {
		let r = n % 4;
		// high leftover rows first, original descending order
		for j in (n - r..n).rev() {
			let col = &a[j * cs..j * cs + n];
			if !unit {
				x[j] /= col[j];
			}
			let t = x[j];
			daxpy_k(-t, &col[..j], &mut x[..j]);
		}
		let mut g = n - r;
		while g >= 4 {
			g -= 4;
			// in-group solve, original descending order (each t depends
			// on the adds from the steps above it)
			let mut t = [0.0f64; 4];
			for u in (0..4).rev() {
				let cj = (g + u) * cs;
				if !unit {
					x[g + u] /= a[cj + g + u];
				}
				t[u] = -x[g + u];
				for i in g..g + u {
					x[i] += a[cj + i] * t[u];
				}
			}
			// common prefix x[..g], sources in descending step order
			unsafe {
				daxpy4in(
					ap.add((g + 3) * cs),
					ap.add((g + 2) * cs),
					ap.add((g + 1) * cs),
					ap.add(g * cs),
					[t[3], t[2], t[1], t[0]],
					x.as_mut_ptr(),
					g,
				);
			}
		}
	} else {
		let mut j = 0usize;
		while j + 4 <= n {
			// in-group solve, original ascending order
			let mut t = [0.0f64; 4];
			for u in 0..4 {
				let cj = (j + u) * cs;
				if !unit {
					x[j + u] /= a[cj + j + u];
				}
				t[u] = -x[j + u];
				for i in j + u + 1..j + 4 {
					x[i] += a[cj + i] * t[u];
				}
			}
			// common suffix x[j+4..], sources in ascending step order
			unsafe {
				daxpy4in(
					ap.add(j * cs + j + 4),
					ap.add((j + 1) * cs + j + 4),
					ap.add((j + 2) * cs + j + 4),
					ap.add((j + 3) * cs + j + 4),
					t,
					x.as_mut_ptr().add(j + 4),
					n - j - 4,
				);
			}
			j += 4;
		}
		while j < n {
			let col = &a[j * cs..j * cs + n];
			if !unit {
				x[j] /= col[j];
			}
			let t = x[j];
			daxpy_k(-t, &col[j + 1..], &mut x[j + 1..]);
			j += 1;
		}
	}
}

/// `x ← op(A)⁻¹·x` with `op(A) = Aᵀ`, A triangular
/// n×n at column stride `cs`; `upper` names the STORED triangle (op(A)
/// holds the other). Per element: `x[j] ← (x[j] − t) / d` with `t` the fold of stored column j's
/// triangle segment against the already solved part of `x`, the fold being the `ddot`
/// kernel's register-parallel order (`src/L1`), so the contract is a
/// deterministic reduction order rather than the sweep's sequence — the
/// tests replay that fold bit for bit. No singularity check.
pub(crate) fn dtrsv_t_k(n: usize, a: &[f64], cs: usize, upper: bool, unit: bool, x: &mut [f64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "dtrsv_t: x length mismatch");
	if upper {
		// stored upper is op(A) lower: forward substitution, each new x[j]
		// from the already solved x[..j]
		for j in 0..n {
			let col = &a[j * cs..j * cs + j];
			let xs = &x[..j];
			let t = ddot_k(col, xs);
			let v = x[j] - t;
			x[j] = if unit { v } else { v / a[j * cs + j] };
		}
	} else {
		// stored lower is op(A) upper: back substitution from x[j + 1..]
		for j in (0..n).rev() {
			let col = &a[j * cs + j + 1..j * cs + n];
			let xs = &x[j + 1..n];
			let t = ddot_k(col, xs);
			let v = x[j] - t;
			x[j] = if unit { v } else { v / a[j * cs + j] };
		}
	}
}
