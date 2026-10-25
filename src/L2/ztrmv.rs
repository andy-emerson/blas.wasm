//! `ztrmv` — triangular matrix × vector, in place: x ← Ax (complex).
//!
//! Implementation: the `dtrmv` shape verbatim — 4-column fan-in
//! column-zaxpy (`kernels::zaxpy4in`, sources in reference column
//! order: ascending for upper, descending for lower), ragged band and
//! diagonal writes scalar in the original step order. Per-element
//! rounding sequence unchanged — bit-for-bit the plain column loop,
//! locked by the test. Transposed/conjugate forms: not built — no
//! consumer yet (explicit gap).

use super::check_mat;
use crate::{Diag, Error, Trans, Uplo};
use crate::c64::C64;
use crate::kernels::zaxpy4in;
use crate::L1::zaxpy_k;

/// `x ← op(A)·x` for a triangular `A` of order `n`, the triangle named by
/// `uplo` stored at leading dimension `lda`, with a unit diagonal when
/// `diag` says so; `x` has `n` elements. `op(A)` is `A`, `Aᵀ` or `Aᴴ`
/// per `trans`.
///
/// The transposed forms are not built yet (`STATUS.md`, todo S2):
/// `trans` other than [`Trans::NoTrans`] returns
/// [`ErrorKind::Unsupported`](crate::ErrorKind::Unsupported) on
/// argument 2.
///
/// Errors: `lda < max(1, n)` (argument 6); `a` or `x` shorter than its
/// extent (arguments 5, 7).
pub fn ztrmv(uplo: Uplo, trans: Trans, diag: Diag, n: usize, a: &[C64], lda: usize, x: &mut [C64]) -> Result<(), Error> {
	const R: &str = "ztrmv";
	if trans != Trans::NoTrans {
		return Err(crate::error::unsupported(R, 2));
	}
	crate::error::mat(R, 5, 6, a.len(), n, n, lda)?;
	crate::error::vec(R, 7, x.len(), n)?;
	ztrmv_k(n, a, lda, uplo.is_upper(), diag.is_unit(), &mut x[..n]);
	Ok(())
}

/// x ← Ax, A triangular n×n at column stride `cs`. `upper` selects the
/// triangle; `unit` treats the diagonal as ones (stored values
/// ignored).
pub(crate) fn ztrmv_k(n: usize, a: &[C64], cs: usize, upper: bool, unit: bool, x: &mut [C64]) {
	check_mat(a.len(), n, n, cs);
	assert_eq!(x.len(), n, "ztrmv: x length mismatch");
	let ap = a.as_ptr();
	if upper {
		let mut j = 0usize;
		while j + 4 <= n {
			// all four t's are original: step j+u writes only x[..=j+u']
			// for u' < u, never x[j+u]
			let t = [x[j], x[j + 1], x[j + 2], x[j + 3]];
			unsafe {
				zaxpy4in(
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
					x[i] = x[i] + t[u] * a[cj + i];
				}
				if !unit {
					x[j + u] = t[u] * a[cj + j + u];
				}
			}
			j += 4;
		}
		while j < n {
			let cj = j * cs;
			let t = x[j];
			zaxpy_k(t, &a[cj..cj + j], &mut x[..j]);
			if !unit {
				x[j] = t * a[cj + j];
			}
			j += 1;
		}
	} else {
		let r = n % 4;
		// high leftover columns first, original descending order
		for j in (n - r..n).rev() {
			let cj = j * cs;
			let t = x[j];
			zaxpy_k(t, &a[cj + j + 1..cj + n], &mut x[j + 1..]);
			if !unit {
				x[j] = t * a[cj + j];
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
				zaxpy4in(
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
					x[i] = x[i] + t[u] * a[cj + i];
				}
				if !unit {
					x[g + u] = t[u] * a[cj + g + u];
				}
			}
		}
	}
}
