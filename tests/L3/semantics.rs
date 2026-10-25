//! Reference BLAS's rules about what is not read: `alpha == 0` (and
//! `k == 0` for the products) leaves A and B unread — C ← βC for the
//! products and rank updates, B ← 0 for trmm/trsm — and `beta == 0`
//! never reads C. Operands that must not be read are poisoned with
//! NaN, so a read would show in the output.
use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::{C32, C64};

#[test]
fn alpha_zero_and_k_zero_scale_c_without_reading_a_or_b() {
	let (m, n, k) = (5, 6, 4);
	let a = vec![f64::NAN; 64];
	let b = vec![f64::NAN; 64];
	let c0: Vec<f64> = (0..m * n).map(|i| i as f64 + 1.0).collect();
	for (alpha, kk) in [(0.0, k), (0.7, 0), (0.0, 0)] {
		for transa in [Trans::NoTrans, Trans::Trans] {
			// the query cannot see α, so the workspace is still owed
			let mut w = vec![f64::NAN; dgemm_work(transa, Trans::NoTrans, m, n, kk)];
			let mut c = c0.clone();
			dgemm(transa, Trans::NoTrans, m, n, kk, alpha, &a, 8, &b, 8, 0.5, &mut c, m, &mut w).unwrap();
			assert_eq!(c, c0.iter().map(|v| v * 0.5).collect::<Vec<_>>(), "dgemm α={alpha} k={kk}");
			let mut c = c0.clone();
			dgemm(transa, Trans::NoTrans, m, n, kk, alpha, &a, 8, &b, 8, 1.0, &mut c, m, &mut w).unwrap();
			assert_eq!(c, c0, "dgemm β = 1 leaves C");
			let mut c = vec![f64::NAN; m * n];
			dgemm(transa, Trans::NoTrans, m, n, kk, alpha, &a, 8, &b, 8, 0.0, &mut c, m, &mut w).unwrap();
			assert_eq!(c, vec![0.0; m * n], "dgemm β = 0 writes zeros without reading C");
		}
	}
	let mut w = vec![f64::NAN; 64];
	let mut c = c0.clone();
	dsymm(Side::Left, Uplo::Upper, m, n, 0.0, &a, 8, &b, 8, 2.0, &mut c, m).unwrap();
	assert_eq!(c, c0.iter().map(|v| v * 2.0).collect::<Vec<_>>());
	// the rank updates scale only the stored triangle
	let nn = 5;
	let cs0: Vec<f64> = (0..nn * nn).map(|i| i as f64 + 1.0).collect();
	for (alpha, kk) in [(0.0, k), (0.7, 0)] {
		for uplo in [Uplo::Upper, Uplo::Lower] {
			let mut c = cs0.clone();
			dsyrk(uplo, Trans::Trans, nn, kk, alpha, &a, 8, 0.5, &mut c, nn, &mut w).unwrap();
			let mut c2 = cs0.clone();
			dsyr2k(uplo, Trans::NoTrans, nn, kk, alpha, &a, 8, &b, 8, 0.5, &mut c2, nn, &mut w).unwrap();
			for j in 0..nn {
				for i in 0..nn {
					let stored = if uplo == Uplo::Upper { i <= j } else { i >= j };
					let want = if stored { cs0[j * nn + i] * 0.5 } else { cs0[j * nn + i] };
					assert_eq!(c[j * nn + i], want, "dsyrk {uplo:?} α={alpha} k={kk} ({i},{j})");
					assert_eq!(c2[j * nn + i], want, "dsyr2k {uplo:?} α={alpha} k={kk} ({i},{j})");
				}
			}
		}
	}
	// trmm/trsm with α = 0: B ← 0, A and B unread
	let mut bb = vec![f64::NAN; m * n];
	dtrmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, m, n, 0.0, &a, 8, &mut bb, m, &mut w).unwrap();
	assert_eq!(bb, vec![0.0; m * n]);
	let mut bb = vec![f64::NAN; m * n];
	dtrsm(Side::Right, Uplo::Upper, Trans::Trans, Diag::Unit, m, n, 0.0, &a, 8, &mut bb, m, &mut w).unwrap();
	assert_eq!(bb, vec![0.0; m * n]);
}

#[test]
fn complex_alpha_zero_rules_and_the_hermitian_diagonal() {
	let (m, n, k) = (4, 3, 5);
	let az = vec![C64::new(f64::NAN, f64::NAN); 64];
	let c0: Vec<C64> = (0..m * n).map(|i| C64::new(i as f64, -1.0)).collect();
	let mut wz = vec![C64::ZERO; 64];
	let mut c = c0.clone();
	zgemm(Trans::ConjTrans, Trans::NoTrans, m, n, k, C64::ZERO, &az, 8, &az, 8, C64::new(0.5, 0.0), &mut c, m, &mut wz).unwrap();
	assert_eq!(c, c0.iter().map(|v| C64::new(v.re * 0.5, v.im * 0.5)).collect::<Vec<_>>());
	let mut c = vec![C64::new(f64::NAN, 0.0); m * n];
	zhemm(Side::Right, Uplo::Lower, m, n, C64::ZERO, &az, 8, &az, 8, C64::ZERO, &mut c, m).unwrap();
	assert_eq!(c, vec![C64::ZERO; m * n]);
	// herk/her2k: real β on the stored triangle, the diagonal left exactly real
	let nn = 4;
	let ch0: Vec<C64> = (0..nn * nn).map(|i| C64::new(i as f64 + 1.0, 0.25)).collect();
	let mut c = ch0.clone();
	zherk(Uplo::Upper, Trans::ConjTrans, nn, k, 0.0, &az, 8, 0.5, &mut c, nn, &mut wz).unwrap();
	let mut c2 = ch0.clone();
	zher2k(Uplo::Lower, Trans::NoTrans, nn, 0, C64::ONE, &az, 8, &az, 8, 0.5, &mut c2, nn, &mut wz).unwrap();
	for j in 0..nn {
		for i in 0..nn {
			let v = ch0[j * nn + i];
			let want_u = if i < j { C64::new(v.re * 0.5, v.im * 0.5) } else if i == j { C64::new(v.re * 0.5, 0.0) } else { v };
			let want_l = if i > j { C64::new(v.re * 0.5, v.im * 0.5) } else if i == j { C64::new(v.re * 0.5, 0.0) } else { v };
			assert_eq!(c[j * nn + i], want_u, "zherk ({i},{j})");
			assert_eq!(c2[j * nn + i], want_l, "zher2k ({i},{j})");
		}
	}
	// with beta == 1 the reference returns before touching C, so even a
	// diagonal with an imaginary part survives, bit for bit
	let ch1: Vec<C64> = (0..nn * nn).map(|i| C64::new(i as f64 + 1.0, 0.25)).collect();
	let mut c = ch1.clone();
	zherk(Uplo::Upper, Trans::NoTrans, nn, 0, 1.0, &az, 8, 1.0, &mut c, nn, &mut wz).unwrap();
	assert_eq!(c, ch1, "zherk k = 0, beta = 1");
	let mut c = ch1.clone();
	zher2k(Uplo::Lower, Trans::ConjTrans, nn, k, C64::ZERO, &az, 8, &az, 8, 1.0, &mut c, nn, &mut wz).unwrap();
	assert_eq!(c, ch1, "zher2k alpha = 0, beta = 1");
	let ch1c: Vec<C32> = ch1.iter().map(|v| C32::new(v.re as f32, v.im as f32)).collect();
	let acn = vec![C32::new(f32::NAN, f32::NAN); 64];
	let mut wc1 = vec![C32::ZERO; 64];
	let mut c = ch1c.clone();
	cherk(Uplo::Lower, Trans::ConjTrans, nn, k, 0.0, &acn, 8, 1.0, &mut c, nn, &mut wc1).unwrap();
	assert_eq!(c, ch1c, "cherk alpha = 0, beta = 1");
	let mut c = ch1c.clone();
	cher2k(Uplo::Upper, Trans::NoTrans, nn, 0, C32::ONE, &acn, 8, &acn, 8, 1.0, &mut c, nn, &mut wc1).unwrap();
	assert_eq!(c, ch1c, "cher2k k = 0, beta = 1");
	let mut b = vec![C32::new(f32::NAN, f32::NAN); m * n];
	let ac = vec![C32::new(f32::NAN, 0.0); 64];
	let mut wc = vec![C32::ZERO; 64];
	ctrsm(Side::Left, Uplo::Upper, Trans::ConjTrans, Diag::NonUnit, m, n, C32::ZERO, &ac, 8, &mut b, m, &mut wc).unwrap();
	assert_eq!(b, vec![C32::ZERO; m * n]);
	let mut cc = vec![C32::new(f32::NAN, 1.0); m * n];
	cgemm(Trans::NoTrans, Trans::NoTrans, m, n, 0, C32::ONE, &ac, 8, &ac, 8, C32::ZERO, &mut cc, m, &mut wc).unwrap();
	assert_eq!(cc, vec![C32::ZERO; m * n]);
}
