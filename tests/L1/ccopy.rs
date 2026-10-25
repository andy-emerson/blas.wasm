use super::common::*;
use blas_wasm::L1::*;
use blas_wasm::C32;

#[test]
fn ccopy_exact() {
	let mut rng = Lcg(104);
	for &n in SIZES {
		let x = rng.vec_c32(n);
		let mut y = vec![C32::new(f32::NAN, f32::NAN); n];
		ccopy(&x, &mut y).unwrap();
		for i in 0..n {
			assert!(bits_eq_cc(y[i], x[i]), "ccopy n={n} i={i}");
		}
	}
}

#[test]
fn ccopy_length_mismatch_is_an_error() {
	let err = ccopy(&[C32::ONE], &mut []).unwrap_err();
	assert_eq!(err, blas_wasm::Error { routine: "ccopy", arg: 2, kind: blas_wasm::ErrorKind::Length });
}
