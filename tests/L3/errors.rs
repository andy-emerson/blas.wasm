//! The argument contract of the Level 3 wrappers: every checkable
//! precondition comes back as an `Error` naming the routine and the
//! argument position, nothing is written when it does, and forms not
//! built yet say so rather than computing something else.
use blas_wasm::L3::*;
use blas_wasm::{Diag, Error, ErrorKind, Side, Trans, Uplo, C64};

fn e(routine: &'static str, arg: u8, kind: ErrorKind) -> Error {
	Error { routine, arg, kind }
}

#[test]
fn gemm_reports_each_bad_argument_by_position() {
	let (m, n, k) = (3, 2, 4);
	let a = vec![1.0; 3 * 4];
	let b = vec![1.0; 4 * 2];
	let mut c = vec![0.0; 3 * 2];
	let mut w: Vec<f64> = vec![];
	let ok = |a: &[f64], lda, b: &[f64], ldb, c: &mut [f64], ldc, w: &mut [f64]| {
		dgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc, w)
	};
	assert_eq!(ok(&a, 2, &b, 4, &mut c, 3, &mut w), Err(e("dgemm", 8, ErrorKind::Argument)));
	assert_eq!(ok(&a[..11], 3, &b, 4, &mut c, 3, &mut w), Err(e("dgemm", 7, ErrorKind::Length)));
	assert_eq!(ok(&a, 3, &b, 3, &mut c, 3, &mut w), Err(e("dgemm", 10, ErrorKind::Argument)));
	assert_eq!(ok(&a, 3, &b[..7], 4, &mut c, 3, &mut w), Err(e("dgemm", 9, ErrorKind::Length)));
	assert_eq!(ok(&a, 3, &b, 4, &mut c, 2, &mut w), Err(e("dgemm", 13, ErrorKind::Argument)));
	assert_eq!(ok(&a, 3, &b, 4, &mut c[..5], 3, &mut w), Err(e("dgemm", 12, ErrorKind::Length)));
	assert_eq!(
		dgemm(Trans::Trans, Trans::NoTrans, m, n, k, 1.0, &a, 4, &b, 4, 0.0, &mut c, 3, &mut w),
		Err(e("dgemm", 1, ErrorKind::Unsupported))
	);
	assert_eq!(
		dgemm(Trans::NoTrans, Trans::ConjTrans, m, n, k, 1.0, &a, 3, &b, 2, 0.0, &mut c, 3, &mut w),
		Err(e("dgemm", 2, ErrorKind::Unsupported))
	);
	// nothing was written by any of the rejected calls; the valid call computes
	assert!(c.iter().all(|&v| v == 0.0));
	assert_eq!(ok(&a, 3, &b, 4, &mut c, 3, &mut w), Ok(()));
	assert!(c.iter().all(|&v| v == 4.0));
}

#[test]
fn gemm_workspace_query_governs_the_packed_shape() {
	// below the packed threshold no scratch is needed
	assert_eq!(dgemm_work(64, 64, 64), 0);
	assert_eq!(sgemm_work(64, 64, 64), 0);
	assert_eq!(zgemm_work(16, 16, 16), 0);
	assert_eq!(cgemm_work(16, 16, 16), 0);
	// above it the query is positive and a short slice is rejected on the work argument
	let n = 520;
	assert!(dgemm_work(n, n, n) > 0);
	let a = vec![0.0; n * n];
	let b = vec![0.0; n * n];
	let mut c = vec![0.0; n * n];
	let mut short = vec![0.0; dgemm_work(n, n, n) - 1];
	assert_eq!(
		dgemm(Trans::NoTrans, Trans::NoTrans, n, n, n, 1.0, &a, n, &b, n, 0.0, &mut c, n, &mut short),
		Err(e("dgemm", 14, ErrorKind::Workspace))
	);
	let mut exact = vec![0.0; dgemm_work(n, n, n)];
	assert_eq!(dgemm(Trans::NoTrans, Trans::NoTrans, n, n, n, 1.0, &a, n, &b, n, 0.0, &mut c, n, &mut exact), Ok(()));
}

#[test]
fn syrk_family_flags_and_dimensions() {
	let n = 3;
	let k = 2;
	let a = vec![1.0; n * k];
	let b = vec![1.0; n * k];
	let mut c = vec![0.0; n * n];
	let mut w: Vec<f64> = vec![];
	assert_eq!(dsyrk(Uplo::Upper, Trans::NoTrans, n, k, 1.0, &a, n, 0.0, &mut c, n, &mut w), Ok(()));
	assert_eq!(dsyrk(Uplo::Upper, Trans::Trans, n, k, 1.0, &a, k, 0.0, &mut c, n, &mut w), Err(e("dsyrk", 2, ErrorKind::Unsupported)));
	assert_eq!(dsyrk(Uplo::Upper, Trans::NoTrans, n, k, 1.0, &a, 2, 0.0, &mut c, n, &mut w), Err(e("dsyrk", 7, ErrorKind::Argument)));
	assert_eq!(dsyrk(Uplo::Upper, Trans::NoTrans, n, k, 1.0, &a, n, 0.0, &mut c, 2, &mut w), Err(e("dsyrk", 10, ErrorKind::Argument)));
	assert_eq!(dsyr2k(Uplo::Lower, Trans::NoTrans, n, k, 1.0, &a, n, &b, n, 0.0, &mut c, n, &mut w), Ok(()));
	assert_eq!(dsyr2k(Uplo::Lower, Trans::NoTrans, n, k, 1.0, &a, n, &b, 2, 0.0, &mut c, n, &mut w), Err(e("dsyr2k", 9, ErrorKind::Argument)));
	// Hermitian: Trans is invalid, ConjTrans is not built yet
	let az = vec![C64::ONE; n * k];
	let mut cz = vec![C64::ZERO; n * n];
	let mut wz: Vec<C64> = vec![];
	assert_eq!(zherk(Uplo::Upper, Trans::Trans, n, k, 1.0, &az, k, 0.0, &mut cz, n, &mut wz), Err(e("zherk", 2, ErrorKind::Argument)));
	assert_eq!(zherk(Uplo::Upper, Trans::ConjTrans, n, k, 1.0, &az, k, 0.0, &mut cz, n, &mut wz), Err(e("zherk", 2, ErrorKind::Unsupported)));
	assert_eq!(zherk(Uplo::Upper, Trans::NoTrans, n, k, 1.0, &az, n, 0.0, &mut cz, n, &mut wz), Ok(()));
}

#[test]
fn symm_and_triangular_sides_size_a_by_side() {
	let (m, n) = (2, 3);
	let a = vec![1.0; 9]; // big enough for either side
	let b = vec![1.0; m * n];
	let mut c = vec![0.0; m * n];
	let mut w: Vec<f64> = vec![];
	assert_eq!(dsymm(Side::Left, Uplo::Upper, m, n, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2), Ok(()));
	assert_eq!(dsymm(Side::Right, Uplo::Upper, m, n, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2), Err(e("dsymm", 7, ErrorKind::Argument)));
	assert_eq!(dsymm(Side::Right, Uplo::Upper, m, n, 1.0, &a, 3, &b, 2, 0.0, &mut c, 2), Ok(()));
	assert_eq!(dsymm(Side::Left, Uplo::Upper, m, n, 1.0, &a, 2, &b, 1, 0.0, &mut c, 2), Err(e("dsymm", 9, ErrorKind::Argument)));
	assert_eq!(dsymm(Side::Left, Uplo::Upper, m, n, 1.0, &a, 2, &b, 2, 0.0, &mut c, 1), Err(e("dsymm", 12, ErrorKind::Argument)));
	let mut bb = b.clone();
	assert_eq!(dtrsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, m, n, 1.0, &a, 2, &mut bb, 2, &mut w), Ok(()));
	assert_eq!(dtrsm(Side::Right, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, m, n, 1.0, &a, 2, &mut bb, 2, &mut w), Err(e("dtrsm", 9, ErrorKind::Argument)));
	assert_eq!(dtrsm(Side::Left, Uplo::Lower, Trans::Trans, Diag::NonUnit, m, n, 1.0, &a, 2, &mut bb, 2, &mut w), Err(e("dtrsm", 3, ErrorKind::Unsupported)));
	assert_eq!(dtrmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::Unit, m, n, 1.0, &a, 2, &mut bb, 1, &mut w), Err(e("dtrmm", 11, ErrorKind::Argument)));
	assert_eq!(dtrmm_work(Side::Left, Trans::NoTrans, m, n), 0);
}

#[test]
fn empty_dimensions_return_ok_and_k_zero_scales_c() {
	// netlib quick returns: m == 0 or n == 0 computes nothing and touches
	// nothing; k == 0 is a legal product of empty operands, C ← βC
	let mut c = vec![2.0; 4];
	let mut w: Vec<f64> = vec![];
	assert_eq!(dgemm(Trans::NoTrans, Trans::NoTrans, 0, 2, 3, 1.0, &[], 1, &[1.0; 6], 3, 0.0, &mut [], 1, &mut w), Ok(()));
	assert_eq!(dgemm(Trans::NoTrans, Trans::NoTrans, 2, 0, 3, 1.0, &[1.0; 6], 2, &[], 3, 0.0, &mut c, 2, &mut w), Ok(()));
	assert_eq!(c, vec![2.0; 4]);
	assert_eq!(dgemm_work(0, 2, 3), 0);
	assert_eq!(dgemm(Trans::NoTrans, Trans::NoTrans, 2, 2, 0, 1.0, &[], 2, &[], 1, 0.5, &mut c, 2, &mut w), Ok(()));
	assert_eq!(c, vec![1.0; 4]);
	let mut cz = vec![C64::new(2.0, 2.0); 4];
	let mut wz: Vec<C64> = vec![];
	assert_eq!(zgemm(Trans::NoTrans, Trans::NoTrans, 0, 2, 3, C64::ONE, &[], 1, &[C64::ONE; 6], 3, C64::ZERO, &mut [], 1, &mut wz), Ok(()));
	assert_eq!(zgemm(Trans::NoTrans, Trans::NoTrans, 2, 2, 0, C64::ONE, &[], 2, &[], 1, C64::new(0.5, 0.0), &mut cz, 2, &mut wz), Ok(()));
	assert_eq!(cz, vec![C64::new(1.0, 1.0); 4]);
	assert_eq!(dsymm(Side::Left, Uplo::Upper, 0, 3, 1.0, &[], 1, &[], 1, 0.0, &mut [], 1), Ok(()));
	assert_eq!(dsymm(Side::Right, Uplo::Upper, 3, 0, 1.0, &[], 1, &[], 3, 0.0, &mut [], 3), Ok(()));
	assert_eq!(dsyrk(Uplo::Upper, Trans::NoTrans, 0, 3, 1.0, &[], 1, 0.0, &mut [], 1, &mut w), Ok(()));
	let mut c3 = vec![2.0; 9];
	assert_eq!(dsyrk(Uplo::Upper, Trans::NoTrans, 3, 0, 1.0, &[], 3, 0.5, &mut c3, 3, &mut w), Ok(()));
	assert_eq!(&c3[..3], &[1.0, 2.0, 2.0]); // upper triangle scaled, the rest untouched
	assert_eq!(dsyr2k(Uplo::Lower, Trans::NoTrans, 0, 3, 1.0, &[], 1, &[], 1, 0.0, &mut [], 1, &mut w), Ok(()));
	assert_eq!(zherk(Uplo::Upper, Trans::NoTrans, 0, 3, 1.0, &[], 1, 0.0, &mut [], 1, &mut wz), Ok(()));
	assert_eq!(dtrmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, 0, 3, 1.0, &[], 1, &mut [], 1, &mut w), Ok(()));
	assert_eq!(dtrmm(Side::Right, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, 3, 0, 1.0, &[], 1, &mut [], 3, &mut w), Ok(()));
	assert_eq!(dtrsm(Side::Right, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, 0, 3, 1.0, &[1.0; 9], 3, &mut [], 1, &mut w), Ok(()));
	assert_eq!(dtrsm(Side::Left, Uplo::Upper, Trans::NoTrans, Diag::Unit, 3, 0, 1.0, &[1.0; 9], 3, &mut [], 3, &mut w), Ok(()));
}
