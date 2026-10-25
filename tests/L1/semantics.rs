//! Reference BLAS's rules about what is not read: `alpha == 0` axpy
//! touches nothing, and iamax's NaN rule. Operands that must not be
//! read are poisoned with NaN, so a read would show in the output.
use blas_wasm::L1::*;
use blas_wasm::{C32, C64};

#[test]
fn axpy_alpha_zero_reads_nothing() {
	let x = [f64::NAN; 5];
	let mut y = [1.0f64, 2.0, 3.0, 4.0, 5.0];
	daxpy(0.0, &x, &mut y).unwrap();
	assert_eq!(y, [1.0, 2.0, 3.0, 4.0, 5.0]);
	daxpy(-0.0, &x, &mut y).unwrap();
	assert_eq!(y, [1.0, 2.0, 3.0, 4.0, 5.0]);
	let xs = [f32::NAN; 3];
	let mut ys = [1.0f32, 2.0, 3.0];
	saxpy(0.0, &xs, &mut ys).unwrap();
	assert_eq!(ys, [1.0, 2.0, 3.0]);
	let xz = [C64::new(f64::NAN, f64::NAN); 3];
	let mut yz = [C64::new(1.0, -1.0); 3];
	zaxpy(C64::ZERO, &xz, &mut yz).unwrap();
	assert_eq!(yz, [C64::new(1.0, -1.0); 3]);
	let xc = [C32::new(f32::NAN, 0.0); 3];
	let mut yc = [C32::new(1.0, -1.0); 3];
	caxpy(C32::new(0.0, -0.0), &xc, &mut yc).unwrap();
	assert_eq!(yc, [C32::new(1.0, -1.0); 3]);
	// a nonzero alpha still propagates
	daxpy(1.0, &x, &mut y).unwrap();
	assert!(y.iter().all(|v| v.is_nan()));
}

#[test]
fn iamax_nan_follows_the_reference_rule() {
	// reference: the first element is the running maximum; a NaN there is
	// never exceeded, so index 0; a NaN elsewhere never compares greater
	let nan = f64::NAN;
	assert_eq!(idamax(&[nan, 1.0, 2.0]), 0);
	assert_eq!(idamax(&[1.0, nan, 2.0]), 2);
	assert_eq!(idamax(&[1.0, 2.0, nan]), 1);
	assert_eq!(idamax(&[3.0, nan, 2.0]), 0);
	let v: Vec<f64> = (0..20).map(|i| if i == 7 { nan } else { i as f64 }).collect();
	assert_eq!(idamax(&v), 19);
	let v: Vec<f64> = (0..20).map(|i| if i == 0 { nan } else { i as f64 }).collect();
	assert_eq!(idamax(&v), 0);
	assert_eq!(isamax(&[f32::NAN, 1.0]), 0);
	assert_eq!(isamax(&[1.0, f32::NAN, 0.5]), 0);
	assert_eq!(izamax(&[C64::new(0.0, nan), C64::ONE]), 0);
	assert_eq!(izamax(&[C64::ONE, C64::new(nan, 0.0), C64::new(2.0, 2.0)]), 2);
	assert_eq!(icamax(&[C32::new(f32::NAN, 0.0), C32::ONE]), 0);
	assert_eq!(icamax(&[C32::ONE, C32::new(0.0, f32::NAN)]), 0);
}
