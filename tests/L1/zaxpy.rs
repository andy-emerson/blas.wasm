use super::common::*;
use blas_wasm::L1::*;
use blas_wasm::C64;

#[test]
fn zaxpy_bit_for_bit() {
	let mut rng = Lcg(101);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let y0 = rng.vec_c64(n);
		for alpha in [
			C64::new(-0.7, 0.3),
			C64::new(1.5, 0.0),
			C64::new(0.0, -1.1),
			C64::ZERO,
		] {
			let mut y = y0.clone();
			zaxpy(alpha, &x, &mut y).unwrap();
			for i in 0..n {
				let want = y0[i] + alpha * x[i];
				assert!(bits_eq_c(y[i], want), "zaxpy n={n} i={i} α={alpha:?}");
			}
		}
	}
}

#[test]
fn zaxpy_length_mismatch_is_an_error() {
	let err = zaxpy(C64::ONE, &[C64::ONE], &mut []).unwrap_err();
	assert_eq!(err, blas_wasm::Error { routine: "zaxpy", arg: 3, kind: blas_wasm::ErrorKind::Length });
}
