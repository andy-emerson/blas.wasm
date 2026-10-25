use super::common::*;
use blas_wasm::L2::*;
use blas_wasm::C64;

#[test]
fn ztrsv_bit_for_bit_and_residual() {
	let mut rng = Lcg(129);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				// diagonally dominant triangle with a genuinely complex
				// diagonal, so Smith's division is exercised
				let mut a = rng.mat_c64(n, n, cs);
				for j in 0..n {
					a[j * cs + j] = C64::new(2.0 * (n as f64) + 1.0 + j as f64, 0.7);
				}
				let b = rng.vec_c64(n);
				let mut x = b.clone();
				ztrsv(uplo(upper), Trans::NoTrans, diag(unit), n, &a, cs, &mut x).unwrap();

				// same-order scalar replay: bit-for-bit
				let mut xr = b.clone();
				if upper {
					for j in (0..n).rev() {
						if !unit {
							xr[j] = xr[j] / a[j * cs + j];
						}
						let t = -xr[j];
						for i in 0..j {
							xr[i] = xr[i] + t * a[j * cs + i];
						}
					}
				} else {
					for j in 0..n {
						if !unit {
							xr[j] = xr[j] / a[j * cs + j];
						}
						let t = -xr[j];
						for i in j + 1..n {
							xr[i] = xr[i] + t * a[j * cs + i];
						}
					}
				}
				for i in 0..n {
					assert!(
						bits_eq_c(x[i], xr[i]),
						"ztrsv bits upper={upper} unit={unit} n={n} i={i}"
					);
				}

				// independent residual: A·x must reproduce b
				let xmax = x.iter().fold(0.0f64, |m, v| m.max(v.abs1()));
				for i in 0..n {
					let ax = comp_sum_c((0..n).map(|j| {
						let in_tri = if upper { i <= j } else { i >= j };
						if !in_tri {
							return C64::ZERO;
						}
						let aij = if unit && i == j { C64::ONE } else { a[j * cs + i] };
						aij * x[j]
					}));
					let scale = xmax * (3.0 * n as f64 + 2.0) + b[i].abs1();
					let tol = f64::EPSILON * (n.max(1) as f64) * 16.0 * scale + 1e-300;
					assert!(
						(ax.re - b[i].re).abs() + (ax.im - b[i].im).abs() <= tol,
						"ztrsv residual upper={upper} unit={unit} n={n} i={i}"
					);
				}
			}
		}
	}
}

#[test]
fn ztrsv_transposed_bit_for_bit_and_against_the_explicit_transpose() {
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
					let mut a = rng.mat_c64(n, n, cs);
					for j in 0..n {
						a[j * cs + j] = C64::new(2.0 * n as f64 + 1.0 + j as f64, 0.5);
					}
					let x0 = rng.vec_c64(n);
					let mut x = x0.clone();
					ztrsv(uplo(upper), trans, diag(unit), n, &a, cs, &mut x).unwrap();
					// same-fold scalar replay: bit-for-bit
					let mut xr = x0.clone();
					let cj = |v: C64| if trans == Trans::ConjTrans { v.conj() } else { v };
					if upper {
						for j in 0..n {
							let col: Vec<C64> = a[j * cs..j * cs + j].to_vec();
							let xs = &xr[..j];
							let tt = dot_fold_c64(&col, xs, trans == Trans::ConjTrans);
							let v = xr[j] - tt;
							xr[j] = if unit { v } else { v / cj(a[j * cs + j]) };
						}
					} else {
						for j in (0..n).rev() {
							let col: Vec<C64> = a[j * cs + j + 1..j * cs + n].to_vec();
							let xs = &xr[j + 1..n];
							let tt = dot_fold_c64(&col, xs, trans == Trans::ConjTrans);
							let v = xr[j] - tt;
							xr[j] = if unit { v } else { v / cj(a[j * cs + j]) };
						}
					}
					for i in 0..n {
						assert!(
							bits_eq_c(x[i], xr[i]),
							"ztrsv {trans:?} upper={upper} unit={unit} n={n} i={i}"
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
					ztrsv(uplo(!upper), Trans::NoTrans, diag(unit), n, &oa, cs2, &mut xe).unwrap();
					let xmax = xe.iter().fold(0.0f64, |m, &v| m.max(v.re.abs() + v.im.abs()));
					let tol = f64::EPSILON * 64.0 * (n as f64 + 1.0) * xmax + 1e-300;
					for i in 0..n {
						assert!(
							((x[i] - xe[i]).re.abs() + (x[i] - xe[i]).im.abs()) <= tol,
							"ztrsv {trans:?} vs explicit transpose upper={upper} unit={unit} n={n} i={i}: {:?} vs {:?}",
							x[i],
							xe[i]
						);
					}
				}
			}
		}
	}
}
