use super::common::*;
use blas_wasm::L1::*;

#[test]
fn axpy_bit_for_bit() {
	let mut rng = Lcg(4);
	for &n in SIZES {
		for alpha in [0.0, 1.0, -2.5, 0.1] {
			let x = rng.vec_f64(n);
			let y0 = rng.vec_f64(n);
			let mut y = y0.clone();
			daxpy(alpha, &x, &mut y).unwrap();
			for i in 0..n {
				let want = y0[i] + x[i] * alpha;
				assert_eq!(y[i].to_bits(), want.to_bits(), "daxpy n={n} i={i}");
			}
		}
	}
}

#[test]
fn daxpy_length_mismatch_is_an_error() {
	let err = daxpy(1.0, &[1.0, 2.0], &mut [1.0]).unwrap_err();
	assert_eq!(err, blas_wasm::Error { routine: "daxpy", arg: 3, kind: blas_wasm::ErrorKind::Length });
}
