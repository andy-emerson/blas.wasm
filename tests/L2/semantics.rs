//! Reference BLAS's rules about what is not read: `alpha == 0` leaves
//! the matrix and x unread (y ← βy for gemv/symv/hemv, nothing for the
//! rank updates), `beta == 0` never reads y. Operands that must not be
//! read are poisoned with NaN, so a read would show in the output.
use super::common::*;
use blas_wasm::L2::*;
use blas_wasm::{C32, C64};

#[test]
fn alpha_zero_reads_neither_matrix_nor_x() {
	let n = 6;
	let a = vec![f64::NAN; n * n];
	let x = vec![f64::NAN; n];
	let y0: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
	for trans in [Trans::NoTrans, Trans::Trans] {
		let mut y = y0.clone();
		dgemv(trans, n, n, 0.0, &a, n, &x, 0.5, &mut y).unwrap();
		assert_eq!(y, y0.iter().map(|v| v * 0.5).collect::<Vec<_>>(), "dgemv {trans:?} β = 0.5");
		let mut y = y0.clone();
		dgemv(trans, n, n, 0.0, &a, n, &x, 1.0, &mut y).unwrap();
		assert_eq!(y, y0, "dgemv {trans:?} β = 1");
		let mut y = vec![f64::NAN; n];
		dgemv(trans, n, n, 0.0, &a, n, &x, 0.0, &mut y).unwrap();
		assert_eq!(y, vec![0.0; n], "dgemv {trans:?} β = 0 writes zeros without reading y");
	}
	let mut y = y0.clone();
	dsymv(Uplo::Upper, n, 0.0, &a, n, &x, 2.0, &mut y).unwrap();
	assert_eq!(y, y0.iter().map(|v| v * 2.0).collect::<Vec<_>>());
	let mut a2 = y0.iter().flat_map(|&v| std::iter::repeat_n(v, n)).collect::<Vec<_>>();
	let keep = a2.clone();
	dger(n, n, 0.0, &x, &x, &mut a2, n).unwrap();
	dsyr(Uplo::Lower, n, 0.0, &x, &mut a2, n).unwrap();
	dsyr2(Uplo::Upper, n, 0.0, &x, &x, &mut a2, n).unwrap();
	assert_eq!(a2, keep);
	// the complex twins
	let az = vec![C64::new(f64::NAN, f64::NAN); n * n];
	let xz = vec![C64::new(f64::NAN, 0.0); n];
	let yz0: Vec<C64> = (0..n).map(|i| C64::new(i as f64, 1.0)).collect();
	let mut yz = yz0.clone();
	zgemv(Trans::ConjTrans, n, n, C64::ZERO, &az, n, &xz, C64::new(2.0, 0.0), &mut yz).unwrap();
	assert_eq!(yz, yz0.iter().map(|v| C64::new(v.re * 2.0, v.im * 2.0)).collect::<Vec<_>>());
	let mut yz = yz0.clone();
	zhemv(Uplo::Lower, n, C64::ZERO, &az, n, &xz, C64::ONE, &mut yz).unwrap();
	assert_eq!(yz, yz0);
	let mut a2z = vec![C64::new(3.0, -3.0); n * n];
	zgeru(n, n, C64::ZERO, &xz, &xz, &mut a2z, n).unwrap();
	zgerc(n, n, C64::ZERO, &xz, &xz, &mut a2z, n).unwrap();
	zher(Uplo::Upper, n, 0.0, &xz, &mut a2z, n).unwrap();
	zher2(Uplo::Lower, n, C64::ZERO, &xz, &xz, &mut a2z, n).unwrap();
	assert_eq!(a2z, vec![C64::new(3.0, -3.0); n * n]);
	let ac = vec![C32::new(f32::NAN, 0.0); n * n];
	let xc = vec![C32::new(0.0, f32::NAN); n];
	let mut yc = vec![C32::new(f32::NAN, f32::NAN); n];
	cgemv(Trans::Trans, n, n, C32::ZERO, &ac, n, &xc, C32::ZERO, &mut yc).unwrap();
	assert_eq!(yc, vec![C32::ZERO; n]);
	let mut ys = vec![f32::NAN; n];
	sgemv(Trans::NoTrans, n, n, 0.0, &vec![f32::NAN; n * n], n, &vec![f32::NAN; n], 0.0, &mut ys).unwrap();
	assert_eq!(ys, vec![0.0f32; n]);
}
