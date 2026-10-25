//! `cgemv` — complex matrix × vector: y ← αAx + βy, with the
//! transpose twin (Aᵀx, unconjugated) and the conjugate-transpose
//! twin (Aᴴx — the form complex algorithms actually consume).
//!
//! Implementation: same shapes as `dgemv` — the no-transpose form is
//! 4-column fan-in column-caxpy (`kernels::caxpy4in`; per-element
//! rounding sequence identical to the plain column loop, bit-for-bit
//! tested), the transpose forms are one `cdotu`/`cdotc` reduction
//! stream per column and inherit the 4-accumulator dot through
//! composition.

use super::{check_mat, cscale_y};
use crate::{Error, Trans};
use crate::c32::C32;
use crate::kernels::caxpy4in;
use crate::L1::{caxpy_k, cdotc_k, cdotu_k};

/// `y ← α·op(A)·x + βy` with `op(A)` = `A`, `Aᵀ` or `Aᴴ` per `trans`.
/// `A` is `m×n` at leading dimension `lda`; `x` has `n` elements and
/// `y` has `m` for [`Trans::NoTrans`], the other way round otherwise.
/// Longer slices are used up to those lengths.
///
/// Errors: `lda < max(1, m)` (argument 6); `a`, `x` or `y` shorter than
/// its extent (arguments 5, 7, 9).
#[allow(clippy::too_many_arguments)]
pub fn cgemv(trans: Trans, m: usize, n: usize, alpha: C32, a: &[C32], lda: usize, x: &[C32], beta: C32, y: &mut [C32]) -> Result<(), Error> {
	const R: &str = "cgemv";
	crate::error::mat(R, 5, 6, a.len(), m, n, lda)?;
	let (xn, yn) = match trans { Trans::NoTrans => (n, m), _ => (m, n) };
	crate::error::vec(R, 7, x.len(), xn)?;
	crate::error::vec(R, 9, y.len(), yn)?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	let (x, y) = (&x[..xn], &mut y[..yn]);
	match trans {
		Trans::NoTrans => cgemv_k(alpha, m, n, a, lda, x, beta, y),
		Trans::Trans => cgemv_t_k(alpha, m, n, a, lda, x, beta, y),
		Trans::ConjTrans => cgemv_c_k(alpha, m, n, a, lda, x, beta, y),
	}
	Ok(())
}

/// y ← αAx + βy. A is nrows×ncols at column stride `cs`;
/// x has ncols elements, y has nrows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cgemv_k(
	alpha: C32,
	nrows: usize,
	ncols: usize,
	a: &[C32],
	cs: usize,
	x: &[C32],
	beta: C32,
	y: &mut [C32],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), ncols, "cgemv: x length mismatch");
	assert_eq!(y.len(), nrows, "cgemv: y length mismatch");
	cscale_y(beta, y);
	let mut j = 0usize;
	while j + 4 <= ncols {
		let t = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
		let ap = a.as_ptr();
		unsafe {
			caxpy4in(
				ap.add(j * cs),
				ap.add((j + 1) * cs),
				ap.add((j + 2) * cs),
				ap.add((j + 3) * cs),
				t,
				y.as_mut_ptr(),
				nrows,
			);
		}
		j += 4;
	}
	while j < ncols {
		caxpy_k(alpha * x[j], &a[j * cs..j * cs + nrows], y);
		j += 1;
	}
}

/// y ← αAᵀx + βy (unconjugated transpose). A is nrows×ncols at column
/// stride `cs`; x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cgemv_t_k(
	alpha: C32,
	nrows: usize,
	ncols: usize,
	a: &[C32],
	cs: usize,
	x: &[C32],
	beta: C32,
	y: &mut [C32],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "cgemv_t: x length mismatch");
	assert_eq!(y.len(), ncols, "cgemv_t: y length mismatch");
	cscale_y(beta, y);
	for j in 0..ncols {
		y[j] = y[j] + alpha * cdotu_k(&a[j * cs..j * cs + nrows], x);
	}
}

/// y ← αAᴴx + βy (conjugate transpose). A is nrows×ncols at column
/// stride `cs`; x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cgemv_c_k(
	alpha: C32,
	nrows: usize,
	ncols: usize,
	a: &[C32],
	cs: usize,
	x: &[C32],
	beta: C32,
	y: &mut [C32],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "cgemv_c: x length mismatch");
	assert_eq!(y.len(), ncols, "cgemv_c: y length mismatch");
	cscale_y(beta, y);
	for j in 0..ncols {
		y[j] = y[j] + alpha * cdotc_k(&a[j * cs..j * cs + nrows], x);
	}
}
