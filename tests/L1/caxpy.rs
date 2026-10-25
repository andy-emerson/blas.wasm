use super::common::*;
use blas_wasm::L1::*;
use blas_wasm::C32;

#[test]
fn caxpy_bit_for_bit() {
	let mut rng = Lcg(101);
	for &n in SIZES {
		let x = rng.vec_c32(n);
		let y0 = rng.vec_c32(n);
		for alpha in [
			C32::new(-0.7, 0.3),
			C32::new(1.5, 0.0),
			C32::new(0.0, -1.1),
			C32::ZERO,
		] {
			let mut y = y0.clone();
			caxpy(alpha, &x, &mut y).unwrap();
			for i in 0..n {
				let want = y0[i] + alpha * x[i];
				assert!(bits_eq_cc(y[i], want), "caxpy n={n} i={i} α={alpha:?}");
			}
		}
	}
}

#[test]
fn caxpy_length_mismatch_is_an_error() {
	let err = caxpy(C32::ONE, &[C32::ONE], &mut []).unwrap_err();
	assert_eq!(err, blas_wasm::Error { routine: "caxpy", arg: 3, kind: blas_wasm::ErrorKind::Length });
}
