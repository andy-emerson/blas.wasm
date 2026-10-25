use super::common::*;
use blas_wasm::L2::*;
use blas_wasm::C64;

#[test]
fn ztrmv_bit_for_bit_all_variants() {
	let mut rng = Lcg(128);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				let a = rng.mat_c64(n, n, cs);
				let x0 = rng.vec_c64(n);
				let mut x = x0.clone();
				ztrmv(uplo(upper), Trans::NoTrans, diag(unit), n, &a, cs, &mut x).unwrap();
				// same-order scalar replay
				let mut xr = x0.clone();
				if upper {
					for j in 0..n {
						let t = xr[j];
						for i in 0..j {
							xr[i] = xr[i] + t * a[j * cs + i];
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				} else {
					for j in (0..n).rev() {
						let t = xr[j];
						for i in j + 1..n {
							xr[i] = xr[i] + t * a[j * cs + i];
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				}
				for i in 0..n {
					assert!(
						bits_eq_c(x[i], xr[i]),
						"ztrmv upper={upper} unit={unit} n={n} i={i}"
					);
				}
			}
		}
	}
}

#[test]
fn ztrmv_transposed_bit_for_bit_and_against_the_explicit_transpose() {
	// the transposed forms reduce with the c64 dot fold per stored
	// column: replayed bit for bit here, and checked against the
	// no-transpose call on an explicitly transposed (conjugated) copy within
	// an n-scaled bound (a different rounding order, so not bitwise)
	let mut rng = Lcg(0x5EC0 + 5);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				for trans in [Trans::Trans, Trans::ConjTrans] {
					let cs = n + 1;
					let a = rng.mat_c64(n, n, cs);
					let x0 = rng.vec_c64(n);
					let mut x = x0.clone();
					ztrmv(uplo(upper), trans, diag(unit), n, &a, cs, &mut x).unwrap();
					// same-fold scalar replay: bit-for-bit
					let mut xr = x0.clone();
					let cj = |v: C64| if trans == Trans::ConjTrans { v.conj() } else { v };
					if upper {
						for j in (0..n).rev() {
							let col: Vec<C64> = a[j * cs..j * cs + j].to_vec();
							let xs = &xr[..j];
							let tt = dot_fold_c64(&col, xs, trans == Trans::ConjTrans);
							let d = if unit { xr[j] } else { xr[j] * cj(a[j * cs + j]) };
							xr[j] = d + tt;
						}
					} else {
						for j in 0..n {
							let col: Vec<C64> = a[j * cs + j + 1..j * cs + n].to_vec();
							let xs = &xr[j + 1..n];
							let tt = dot_fold_c64(&col, xs, trans == Trans::ConjTrans);
							let d = if unit { xr[j] } else { xr[j] * cj(a[j * cs + j]) };
							xr[j] = d + tt;
						}
					}
					for i in 0..n {
						assert!(
							bits_eq_c(x[i], xr[i]),
							"ztrmv {trans:?} upper={upper} unit={unit} n={n} i={i}"
						);
					}
					// the explicit transpose, no-transpose call, triangle flipped
					let cs2 = n + 2;
					let mut oa = vec![C64::default(); cs2 * n.max(1)];
					for j in 0..n {
						for i in 0..n {
							let v = a[i * cs + j];
							oa[j * cs2 + i] = if trans == Trans::ConjTrans { v.conj() } else { v };
						}
					}
					let mut xe = x0.clone();
					ztrmv(uplo(!upper), Trans::NoTrans, diag(unit), n, &oa, cs2, &mut xe).unwrap();
					let xmax = xe.iter().fold(0.0f64, |m, &v| m.max(v.re.abs() + v.im.abs()));
					let tol = f64::EPSILON * 64.0 * (n as f64 + 1.0) * xmax + 1e-300;
					for i in 0..n {
						assert!(
							((x[i] - xe[i]).re.abs() + (x[i] - xe[i]).im.abs()) <= tol,
							"ztrmv {trans:?} vs explicit transpose upper={upper} unit={unit} n={n} i={i}: {:?} vs {:?}",
							x[i],
							xe[i]
						);
					}
				}
			}
		}
	}
}
