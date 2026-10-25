use super::common::*;
use blas_wasm::L1::*;
use blas_wasm::C64;

#[test]
fn zrot_bit_for_bit_the_definition_and_the_real_sine_form() {
	// the scalar complex definition per element, and with a real sine the
	// same bits as the real-rotation routine
	let mut rng = Lcg(0x707 + 1);
	let c: f64 = 0.8;
	let s = C64::new(-0.6, 0.3);
	for &n in SIZES {
		let x0 = rng.vec_c64(n);
		let y0 = rng.vec_c64(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		zrot(&mut x, &mut y, c, s).unwrap();
		for i in 0..n {
			let wx = x0[i].scale(c) + s * y0[i];
			let wy = y0[i].scale(c) - s.conj() * x0[i];
			assert!(bits_eq_c(x[i], wx), "zrot x n={n} i={i}");
			assert!(bits_eq_c(y[i], wy), "zrot y n={n} i={i}");
		}
		let sr = C64::new(-0.6, 0.0);
		let (mut x1, mut y1) = (x0.clone(), y0.clone());
		zrot(&mut x1, &mut y1, c, sr).unwrap();
		let (mut x2, mut y2) = (x0.clone(), y0.clone());
		zdrot(&mut x2, &mut y2, c, -0.6).unwrap();
		for i in 0..n {
			assert!(bits_eq_c(x1[i], x2[i]) && bits_eq_c(y1[i], y2[i]), "zrot real sine vs zdrot n={n} i={i}");
		}
	}
	assert_eq!(zrot(&mut [C64::ONE], &mut [], c, s), Err(blas_wasm::Error { routine: "zrot", arg: 2, kind: blas_wasm::ErrorKind::Length }));
}
