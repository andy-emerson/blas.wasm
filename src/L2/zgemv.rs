//! `zgemv` — complex matrix × vector: y ← αAx + βy, with the
//! transpose twin (Aᵀx, unconjugated) and the conjugate-transpose
//! twin (Aᴴx — the form complex algorithms actually consume).
//!
//! Implementation: same shapes as `dgemv` — the no-transpose form is
//! 4-column fan-in column-zaxpy (`kernels::zaxpy4in`; per-element
//! rounding sequence identical to the plain column loop, bit-for-bit
//! tested), the transpose forms are one `zdotu`/`zdotc` reduction
//! stream per column and inherit the 4-accumulator dot through
//! composition.

use super::{check_mat, zscale_y};
use crate::{Error, Trans};
use crate::c64::C64;
use crate::kernels::zaxpy4in;
use crate::L1::{zaxpy_k, zdotc_k, zdotu_k};

/// `y ← α·op(A)·x + βy` with `op(A)` = `A`, `Aᵀ` or `Aᴴ` per `trans`.
/// `A` is `m×n` at leading dimension `lda`; `x` has `n` elements and
/// `y` has `m` for [`Trans::NoTrans`], the other way round otherwise.
/// Longer slices are used up to those lengths.
///
/// Errors: `lda < max(1, m)` (argument 6); `a`, `x` or `y` shorter than
/// its extent (arguments 5, 7, 9).
#[allow(clippy::too_many_arguments)]
pub fn zgemv(trans: Trans, m: usize, n: usize, alpha: C64, a: &[C64], lda: usize, x: &[C64], beta: C64, y: &mut [C64]) -> Result<(), Error> {
	const R: &str = "zgemv";
	crate::error::mat(R, 5, 6, a.len(), m, n, lda)?;
	let (xn, yn) = match trans { Trans::NoTrans => (n, m), _ => (m, n) };
	crate::error::vec(R, 7, x.len(), xn)?;
	crate::error::vec(R, 9, y.len(), yn)?;
	if m == 0 || n == 0 {
		return Ok(());
	}
	let (x, y) = (&x[..xn], &mut y[..yn]);
	match trans {
		Trans::NoTrans => zgemv_k(alpha, m, n, a, lda, x, beta, y),
		Trans::Trans => zgemv_t_k(alpha, m, n, a, lda, x, beta, y),
		Trans::ConjTrans => zgemv_c_k(alpha, m, n, a, lda, x, beta, y),
	}
	Ok(())
}

/// y ← αAx + βy. A is nrows×ncols at column stride `cs`;
/// x has ncols elements, y has nrows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zgemv_k(
	alpha: C64,
	nrows: usize,
	ncols: usize,
	a: &[C64],
	cs: usize,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), ncols, "zgemv: x length mismatch");
	assert_eq!(y.len(), nrows, "zgemv: y length mismatch");
	zscale_y(beta, y);
	let mut j = 0usize;
	while j + 4 <= ncols {
		let t = [alpha * x[j], alpha * x[j + 1], alpha * x[j + 2], alpha * x[j + 3]];
		let ap = a.as_ptr();
		unsafe {
			zaxpy4in(
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
		zaxpy_k(alpha * x[j], &a[j * cs..j * cs + nrows], y);
		j += 1;
	}
}

/// y ← αAᵀx + βy (unconjugated transpose). A is nrows×ncols at column
/// stride `cs`; x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zgemv_t_k(
	alpha: C64,
	nrows: usize,
	ncols: usize,
	a: &[C64],
	cs: usize,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "zgemv_t: x length mismatch");
	assert_eq!(y.len(), ncols, "zgemv_t: y length mismatch");
	zscale_y(beta, y);
	for j in 0..ncols {
		y[j] = y[j] + alpha * zdotu_k(&a[j * cs..j * cs + nrows], x);
	}
}

/// y ← αAᴴx + βy (conjugate transpose). A is nrows×ncols at column
/// stride `cs`; x has nrows elements, y has ncols.
#[allow(clippy::too_many_arguments)]
pub(crate) fn zgemv_c_k(
	alpha: C64,
	nrows: usize,
	ncols: usize,
	a: &[C64],
	cs: usize,
	x: &[C64],
	beta: C64,
	y: &mut [C64],
) {
	check_mat(a.len(), nrows, ncols, cs);
	assert_eq!(x.len(), nrows, "zgemv_c: x length mismatch");
	assert_eq!(y.len(), ncols, "zgemv_c: y length mismatch");
	zscale_y(beta, y);
	for j in 0..ncols {
		y[j] = y[j] + alpha * zdotc_k(&a[j * cs..j * cs + nrows], x);
	}
}
