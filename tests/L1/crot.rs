use super::common::*;
use blas_wasm::L1::*;
use blas_wasm::C32;

#[test]
fn crot_bit_for_bit_the_definition_and_the_real_sine_form() {
	// the scalar complex definition per element, and with a real sine the
	// same bits as the real-rotation routine
	let mut rng = Lcg(0x707 + 1);
	let c: f32 = 0.8;
	let s = C32::new(-0.6, 0.3);
	for &n in SIZES {
		let x0 = rng.vec_c32(n);
		let y0 = rng.vec_c32(n);
		let mut x = x0.clone();
		let mut y = y0.clone();
		crot(&mut x, &mut y, c, s).unwrap();
		for i in 0..n {
			let wx = x0[i].scale(c) + s * y0[i];
			let wy = y0[i].scale(c) - s.conj() * x0[i];
			assert!(bits_eq_cc(x[i], wx), "crot x n={n} i={i}");
			assert!(bits_eq_cc(y[i], wy), "crot y n={n} i={i}");
		}
		let sr = C32::new(-0.6, 0.0);
		let (mut x1, mut y1) = (x0.clone(), y0.clone());
		crot(&mut x1, &mut y1, c, sr).unwrap();
		let (mut x2, mut y2) = (x0.clone(), y0.clone());
		csrot(&mut x2, &mut y2, c, -0.6).unwrap();
		for i in 0..n {
			assert!(bits_eq_cc(x1[i], x2[i]) && bits_eq_cc(y1[i], y2[i]), "crot real sine vs csrot n={n} i={i}");
		}
	}
	assert_eq!(crot(&mut [C32::ONE], &mut [], c, s), Err(blas_wasm::Error { routine: "crot", arg: 2, kind: blas_wasm::ErrorKind::Length }));
}
