use super::common::*;
use blas_wasm::L1::sasum;
use blas_wasm::L2::*;

#[test]
fn trsv_bit_for_bit_and_residual() {
	let mut rng = Lcg(27);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				// diagonally dominant triangle: solves stay well-conditioned
				let mut a = rng.mat_f32(n, n, cs);
				for j in 0..n {
					a[j * cs + j] = 2.0 * (n as f32) + 1.0 + j as f32;
				}
				let b = rng.vec_f32(n);
				let mut x = b.clone();
				strsv(uplo(upper), Trans::NoTrans, diag(unit), n, &a, cs, &mut x).unwrap();

				// same-order scalar replay: bit-for-bit
				let mut xr = b.clone();
				if upper {
					for j in (0..n).rev() {
						if !unit {
							xr[j] /= a[j * cs + j];
						}
						let t = xr[j];
						for i in 0..j {
							xr[i] += a[j * cs + i] * -t;
						}
					}
				} else {
					for j in 0..n {
						if !unit {
							xr[j] /= a[j * cs + j];
						}
						let t = xr[j];
						for i in j + 1..n {
							xr[i] += a[j * cs + i] * -t;
						}
					}
				}
				for i in 0..n {
					assert_eq!(
						x[i].to_bits(),
						xr[i].to_bits(),
						"strsv bits upper={upper} unit={unit} n={n} i={i}"
					);
				}

				// independent residual: A·x must reproduce b
				for i in 0..n {
					let ax = comp_sum32((0..n).map(|j| {
						let in_tri = if upper { i <= j } else { i >= j };
						if !in_tri {
							return 0.0;
						}
						let aij = if unit && i == j { 1.0 } else { a[j * cs + i] };
						aij * x[j]
					}));
					let scale = sasum(&x) as f64 * (2.0 * n as f64 + n as f64) + b[i].abs() as f64;
					let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 8.0 * scale + 1e-40;
					assert!(
						(ax - b[i] as f64).abs() <= tol,
						"strsv residual upper={upper} unit={unit} n={n} i={i}: {ax} vs {}",
						b[i]
					);
				}
			}
		}
	}
}

#[test]
fn strsv_transposed_bit_for_bit_and_against_the_explicit_transpose() {
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
					let mut a = rng.mat_f32(n, n, cs);
					for j in 0..n {
						a[j * cs + j] = 2.0 * (n as f32) + 1.0 + j as f32;
					}
					let x0 = rng.vec_f32(n);
					let mut x = x0.clone();
					strsv(uplo(upper), trans, diag(unit), n, &a, cs, &mut x).unwrap();
					// same-fold scalar replay: bit-for-bit
					let mut xr = x0.clone();
					if upper {
						for j in 0..n {
							let col: Vec<f32> = a[j * cs..j * cs + j].to_vec();
							let xs = &xr[..j];
							let tt = dot_fold_f32(&col, xs);
							let v = xr[j] - tt;
							xr[j] = if unit { v } else { v / a[j * cs + j] };
						}
					} else {
						for j in (0..n).rev() {
							let col: Vec<f32> = a[j * cs + j + 1..j * cs + n].to_vec();
							let xs = &xr[j + 1..n];
							let tt = dot_fold_f32(&col, xs);
							let v = xr[j] - tt;
							xr[j] = if unit { v } else { v / a[j * cs + j] };
						}
					}
					for i in 0..n {
						assert!(
							x[i].to_bits() == xr[i].to_bits(),
							"strsv {trans:?} upper={upper} unit={unit} n={n} i={i}"
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
					strsv(uplo(!upper), Trans::NoTrans, diag(unit), n, &oa, cs2, &mut xe).unwrap();
					let xmax = xe.iter().fold(0.0f64, |m, &v| m.max(v.abs() as f64));
					let tol = f32::EPSILON as f64 * 64.0 * (n as f64 + 1.0) * xmax + 1e-300;
					for i in 0..n {
						assert!(
							(x[i] - xe[i]).abs() as f64 <= tol,
							"strsv {trans:?} vs explicit transpose upper={upper} unit={unit} n={n} i={i}: {:?} vs {:?}",
							x[i],
							xe[i]
						);
					}
				}
			}
		}
	}
}
