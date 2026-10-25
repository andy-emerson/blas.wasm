use super::common::*;
use blas_wasm::L2::*;

#[test]
fn trmv_bit_for_bit_all_variants() {
	let mut rng = Lcg(26);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				let a = rng.mat_f32(n, n, cs);
				let x0 = rng.vec_f32(n);
				let mut x = x0.clone();
				strmv(uplo(upper), Trans::NoTrans, diag(unit), n, &a, cs, &mut x).unwrap();
				// same-order scalar replay
				let mut xr = x0.clone();
				if upper {
					for j in 0..n {
						let t = xr[j];
						for i in 0..j {
							xr[i] += a[j * cs + i] * t;
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				} else {
					for j in (0..n).rev() {
						let t = xr[j];
						for i in j + 1..n {
							xr[i] += a[j * cs + i] * t;
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				}
				for i in 0..n {
					assert_eq!(
						x[i].to_bits(),
						xr[i].to_bits(),
						"strmv upper={upper} unit={unit} n={n} i={i}"
					);
				}
			}
		}
	}
}

#[test]
fn strmv_transposed_bit_for_bit_and_against_the_explicit_transpose() {
	// the transposed forms reduce with the f32 dot fold per stored
	// column: replayed bit for bit here, and checked against the
	// no-transpose call on an explicitly transposed copy within
	// an n-scaled bound (a different rounding order, so not bitwise)
	let mut rng = Lcg(0x5EC0 + 5);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				for trans in [Trans::Trans, Trans::ConjTrans] {
					let cs = n + 1;
					let a = rng.mat_f32(n, n, cs);
					let x0 = rng.vec_f32(n);
					let mut x = x0.clone();
					strmv(uplo(upper), trans, diag(unit), n, &a, cs, &mut x).unwrap();
					// same-fold scalar replay: bit-for-bit
					let mut xr = x0.clone();
					if upper {
						for j in (0..n).rev() {
							let col: Vec<f32> = a[j * cs..j * cs + j].to_vec();
							let xs = &xr[..j];
							let tt = dot_fold_f32(&col, xs);
							let d = if unit { xr[j] } else { xr[j] * a[j * cs + j] };
							xr[j] = d + tt;
						}
					} else {
						for j in 0..n {
							let col: Vec<f32> = a[j * cs + j + 1..j * cs + n].to_vec();
							let xs = &xr[j + 1..n];
							let tt = dot_fold_f32(&col, xs);
							let d = if unit { xr[j] } else { xr[j] * a[j * cs + j] };
							xr[j] = d + tt;
						}
					}
					for i in 0..n {
						assert!(
							x[i].to_bits() == xr[i].to_bits(),
							"strmv {trans:?} upper={upper} unit={unit} n={n} i={i}"
						);
					}
					// the explicit transpose, no-transpose call, triangle flipped
					let cs2 = n + 2;
					let mut oa = vec![f32::default(); cs2 * n.max(1)];
					for j in 0..n {
						for i in 0..n {
							let v = a[i * cs + j];
							oa[j * cs2 + i] = v;
						}
					}
					let mut xe = x0.clone();
					strmv(uplo(!upper), Trans::NoTrans, diag(unit), n, &oa, cs2, &mut xe).unwrap();
					let xmax = xe.iter().fold(0.0f64, |m, &v| m.max(v.abs() as f64));
					let tol = f32::EPSILON as f64 * 64.0 * (n as f64 + 1.0) * xmax + 1e-300;
					for i in 0..n {
						assert!(
							(x[i] - xe[i]).abs() as f64 <= tol,
							"strmv {trans:?} vs explicit transpose upper={upper} unit={unit} n={n} i={i}: {:?} vs {:?}",
							x[i],
							xe[i]
						);
					}
				}
			}
		}
	}
}
