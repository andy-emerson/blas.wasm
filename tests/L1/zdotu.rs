use super::common::*;
use blas_wasm::L1::*;

#[test]
fn zdotu_bounded() {
	let mut rng = Lcg(107);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let y = rng.vec_c64(n);
		let s = zdotu(&x, &y).unwrap();
		// products formed in C64 as the implementation forms them,
		// accumulated component-wise with compensation
		let want = comp_sum_c((0..n).map(|i| x[i] * y[i]));
		let scale = comp_scale_c((0..n).map(|i| x[i] * y[i]));
		let tol = f64::EPSILON * (n.max(1) as f64) * 4.0 * scale + 1e-300;
		assert!((s.re - want.re).abs() <= tol, "zdotu re n={n}");
		assert!((s.im - want.im).abs() <= tol, "zdotu im n={n}");
	}
}

#[test]
fn zdotu_length_mismatch_is_an_error() {
	let err = zdotu(&[blas_wasm::C64::ONE], &[]).unwrap_err();
	assert_eq!(err, blas_wasm::Error { routine: "zdotu", arg: 2, kind: blas_wasm::ErrorKind::Length });
}
