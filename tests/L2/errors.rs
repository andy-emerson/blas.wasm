//! The argument contract of the Level 2 wrappers: every checkable
//! precondition comes back as an `Error` naming the routine and the
//! argument position; longer slices are used up to the declared
//! lengths; forms not built yet say so.
use blas_wasm::L2::*;
use blas_wasm::{Diag, Error, ErrorKind, Trans, Uplo, C64};

fn e(routine: &'static str, arg: u8, kind: ErrorKind) -> Error {
	Error { routine, arg, kind }
}

#[test]
fn gemv_positions_and_the_transposed_roles() {
	let (m, n) = (3, 2);
	let a = vec![1.0; m * n];
	let x = vec![1.0; 4];
	let mut y = vec![0.0; 4];
	assert_eq!(dgemv(Trans::NoTrans, m, n, 1.0, &a, m, &x[..n], 0.0, &mut y[..m]), Ok(()));
	// longer slices are used up to n and m
	assert_eq!(dgemv(Trans::NoTrans, m, n, 1.0, &a, m, &x, 0.0, &mut y), Ok(()));
	assert_eq!(y[3], 0.0, "beyond m nothing is written");
	assert_eq!(dgemv(Trans::NoTrans, m, n, 1.0, &a, 2, &x, 0.0, &mut y), Err(e("dgemv", 6, ErrorKind::Argument)));
	assert_eq!(dgemv(Trans::NoTrans, m, n, 1.0, &a[..5], m, &x, 0.0, &mut y), Err(e("dgemv", 5, ErrorKind::Length)));
	assert_eq!(dgemv(Trans::NoTrans, m, n, 1.0, &a, m, &x[..1], 0.0, &mut y), Err(e("dgemv", 7, ErrorKind::Length)));
	assert_eq!(dgemv(Trans::NoTrans, m, n, 1.0, &a, m, &x, 0.0, &mut y[..2]), Err(e("dgemv", 9, ErrorKind::Length)));
	// transposed: x has m elements and y has n
	assert_eq!(dgemv(Trans::Trans, m, n, 1.0, &a, m, &x[..m], 0.0, &mut y[..n]), Ok(()));
	assert_eq!(dgemv(Trans::Trans, m, n, 1.0, &a, m, &x[..n], 0.0, &mut y[..n]), Err(e("dgemv", 7, ErrorKind::Length)));
	// conjugate transpose exists for complex and equals the transpose for real
	let az = vec![C64::ONE; m * n];
	let xz = vec![C64::ONE; m];
	let mut yz = vec![C64::ZERO; n];
	assert_eq!(zgemv(Trans::ConjTrans, m, n, C64::ONE, &az, m, &xz, C64::ZERO, &mut yz), Ok(()));
	assert_eq!(dgemv(Trans::ConjTrans, m, n, 1.0, &a, m, &x[..m], 0.0, &mut y[..n]), Ok(()));
}

#[test]
fn rank_updates_and_symmetric_positions() {
	let (m, n) = (3, 2);
	let x = vec![1.0; 3];
	let y = vec![1.0; 3];
	let mut a = vec![0.0; 9];
	assert_eq!(dger(m, n, 1.0, &x, &y, &mut a, 3), Ok(()));
	assert_eq!(dger(m, n, 1.0, &x[..2], &y, &mut a, 3), Err(e("dger", 4, ErrorKind::Length)));
	assert_eq!(dger(m, n, 1.0, &x, &y[..1], &mut a, 3), Err(e("dger", 5, ErrorKind::Length)));
	assert_eq!(dger(m, n, 1.0, &x, &y, &mut a, 2), Err(e("dger", 7, ErrorKind::Argument)));
	let mut yy = vec![0.0; 3];
	assert_eq!(dsymv(Uplo::Upper, 3, 1.0, &a, 3, &x, 0.0, &mut yy), Ok(()));
	assert_eq!(dsymv(Uplo::Upper, 3, 1.0, &a, 2, &x, 0.0, &mut yy), Err(e("dsymv", 5, ErrorKind::Argument)));
	assert_eq!(dsyr(Uplo::Lower, 3, 1.0, &x, &mut a, 2), Err(e("dsyr", 6, ErrorKind::Argument)));
	assert_eq!(dsyr2(Uplo::Lower, 3, 1.0, &x, &y, &mut a, 2), Err(e("dsyr2", 7, ErrorKind::Argument)));
	assert_eq!(dsyr2(Uplo::Lower, 3, 1.0, &x, &y[..2], &mut a, 3), Err(e("dsyr2", 5, ErrorKind::Length)));
}

#[test]
fn triangular_vectors_transposes_are_not_built_yet() {
	let n = 3;
	let a = vec![1.0; 9];
	let mut x = vec![1.0; 3];
	assert_eq!(dtrmv(Uplo::Upper, Trans::NoTrans, Diag::NonUnit, n, &a, 3, &mut x), Ok(()));
	assert_eq!(dtrmv(Uplo::Upper, Trans::Trans, Diag::NonUnit, n, &a, 3, &mut x), Err(e("dtrmv", 2, ErrorKind::Unsupported)));
	assert_eq!(dtrsv(Uplo::Upper, Trans::ConjTrans, Diag::Unit, n, &a, 3, &mut x), Err(e("dtrsv", 2, ErrorKind::Unsupported)));
	assert_eq!(dtrsv(Uplo::Upper, Trans::NoTrans, Diag::Unit, n, &a, 2, &mut x), Err(e("dtrsv", 6, ErrorKind::Argument)));
	assert_eq!(dtrsv(Uplo::Upper, Trans::NoTrans, Diag::Unit, n, &a, 3, &mut x[..2]), Err(e("dtrsv", 7, ErrorKind::Length)));
}
