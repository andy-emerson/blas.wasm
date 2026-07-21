use super::common::*;
use blas_wasm::L1::*;
use blas_wasm::C64;

#[test]
fn zcopy_exact() {
	let mut rng = Lcg(104);
	for &n in SIZES {
		let x = rng.vec_c64(n);
		let mut y = vec![C64::new(f64::NAN, f64::NAN); n];
		zcopy(&x, &mut y);
		for i in 0..n {
			assert!(bits_eq_c(y[i], x[i]), "zcopy n={n} i={i}");
		}
	}
}

#[test]
#[should_panic(expected = "length mismatch")]
fn zcopy_length_mismatch_panics() {
	zcopy(&[C64::ONE], &mut []);
}
