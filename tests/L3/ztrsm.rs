use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::C64;

#[test]
fn ztrsm_both_sides_residual() {
	let mut rng = Lcg(138);
	let alpha = C64::new(0.9, 0.3);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				// dominant complex diagonal exercises Smith division
				let mut a = rng.mat_c64(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = C64::new(2.0 * (n as f64) + 1.0 + j as f64, 0.7);
				}
				let bcs = m + 1;
				let b0 = rng.mat_c64(m, n, bcs);
				let tri_at = |i: usize, j: usize| -> C64 {
					let in_tri = if upper { i <= j } else { i >= j };
					if !in_tri {
						C64::ZERO
					} else if unit && i == j {
						C64::ONE
					} else {
						a[j * acs + i]
					}
				};
				let bound = 4.0 * (n as f64 + 1.0);

				// left: A·X = α·B0
				let mut x = b0.clone();
				ztrsm(Side::Left, uplo(upper), Trans::NoTrans, diag(unit), m, n, alpha, &a, acs, &mut x, bcs, &mut vec![C64::ZERO; ztrsm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
				for j in 0..n {
					let xmax =
						(0..n).fold(0.0f64, |acc, l| acc.max(x[j * bcs + l].abs1()));
					for i in 0..m {
						let ax = comp_sum_c((0..n).map(|l| tri_at(i, l) * x[j * bcs + l]));
						let want = alpha * b0[j * bcs + i];
						let tol =
							f64::EPSILON * (n.max(1) as f64) * 16.0 * bound * xmax + 1e-300;
						assert!(
							(ax.re - want.re).abs() + (ax.im - want.im).abs() <= tol,
							"ztrsm_left residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				// right: X·A = α·B0
				let mut x = b0.clone();
				ztrsm(Side::Right, uplo(upper), Trans::NoTrans, diag(unit), m, n, alpha, &a, acs, &mut x, bcs, &mut vec![C64::ZERO; ztrsm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
				for j in 0..n {
					for i in 0..m {
						let xa = comp_sum_c((0..n).map(|l| x[l * bcs + i] * tri_at(l, j)));
						let want = alpha * b0[j * bcs + i];
						let xmax =
							(0..n).fold(0.0f64, |acc, l| acc.max(x[l * bcs + i].abs1()));
						let tol =
							f64::EPSILON * (n.max(1) as f64) * 16.0 * bound * xmax + 1e-300;
						assert!(
							(xa.re - want.re).abs() + (xa.im - want.im).abs() <= tol,
							"ztrsm_right residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn ztrsm_bit_replay_both_sides() {
	let mut rng = Lcg(139);
	let alpha = C64::new(0.9, 0.3);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- α·inv(A)·B, plain αscal-then-ztrsv replay
				let acs = m + 1;
				let mut a = rng.mat_c64(m, m, acs);
				for j in 0..m {
					a[j * acs + j] = C64::new(2.0 * (m as f64) + 1.0 + j as f64, 0.7);
				}
				let b0 = rng.mat_c64(m, n, bcs);
				let mut bt = b0.clone();
				ztrsm(Side::Left, uplo(upper), Trans::NoTrans, diag(unit), m, n, alpha, &a, acs, &mut bt, bcs, &mut vec![C64::ZERO; ztrsm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
				let mut br = b0.clone();
				for j in 0..n {
					for i in 0..m {
						br[j * bcs + i] = alpha * br[j * bcs + i];
					}
					if upper {
						for l in (0..m).rev() {
							if !unit {
								br[j * bcs + l] = br[j * bcs + l] / a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] = br[j * bcs + i] + t * a[l * acs + i];
							}
						}
					} else {
						for l in 0..m {
							if !unit {
								br[j * bcs + l] = br[j * bcs + l] / a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] = br[j * bcs + i] + t * a[l * acs + i];
							}
						}
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.re.is_nan() {
						assert!(bits_eq_c(*x, *y), "ztrsm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- α·B·inv(A)
				let acs = n + 1;
				let mut a = rng.mat_c64(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = C64::new(2.0 * (n as f64) + 1.0 + j as f64, 0.7);
				}
				let b0 = rng.mat_c64(m, n, bcs);
				let mut bt = b0.clone();
				ztrsm(Side::Right, uplo(upper), Trans::NoTrans, diag(unit), m, n, alpha, &a, acs, &mut bt, bcs, &mut vec![C64::ZERO; ztrsm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
				let mut br = b0.clone();
				let elim = |br: &mut Vec<C64>, j: usize, k: usize| {
					let t = -a[j * acs + k];
					for i in 0..m {
						br[j * bcs + i] = br[j * bcs + i] + t * br[k * bcs + i];
					}
				};
				let finish = |br: &mut Vec<C64>, j: usize| {
					if !unit {
						let s = C64::ONE / a[j * acs + j];
						for i in 0..m {
							br[j * bcs + i] = s * br[j * bcs + i];
						}
					}
				};
				if upper {
					// plain ascending replay — the grouping keeps upper
					// fully ascending, so this asserts bit-identity to it
					for j in 0..n {
						for i in 0..m {
							br[j * bcs + i] = alpha * br[j * bcs + i];
						}
						for k in 0..j {
							elim(&mut br, j, k);
						}
						finish(&mut br, j);
					}
				} else {
					// grouped replay: out-of-group solved columns first,
					// then in-group elimination descending
					let r = n % 4;
					let mut gs = n;
					while gs >= r + 4 {
						gs -= 4;
						for u in 0..4 {
							for i in 0..m {
								br[(gs + u) * bcs + i] = alpha * br[(gs + u) * bcs + i];
							}
						}
						for k in gs + 4..n {
							for u in 0..4 {
								elim(&mut br, gs + u, k);
							}
						}
						for tc in (gs..gs + 4).rev() {
							for k in tc + 1..gs + 4 {
								elim(&mut br, tc, k);
							}
							finish(&mut br, tc);
						}
					}
					for j in (0..r).rev() {
						for i in 0..m {
							br[j * bcs + i] = alpha * br[j * bcs + i];
						}
						for k in j + 1..n {
							elim(&mut br, j, k);
						}
						finish(&mut br, j);
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.re.is_nan() {
						assert!(bits_eq_c(*x, *y), "ztrsm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}

#[test]
fn ztrsm_transposed_candidates_equal_the_explicit_transpose_bitwise() {
	// the definition of the flag: op(A) through `transa` is bit-identical
	// to the no-transpose call on an explicitly transposed (conjugated)
	// copy with the triangle flipped. Both candidates (copy, stream) and
	// the routed wrapper, every side/uplo/diag, several shapes including
	// row counts off the kernels' block sizes, workspace NaN-filled.
	let mut rng = Lcg(0x7A11 + 5);
	for (m, n) in [(1, 1), (2, 3), (3, 2), (4, 4), (5, 7), (7, 5), (8, 9), (9, 8), (17, 13), (33, 6), (6, 33), (64, 65)] {
		for side in [Side::Left, Side::Right] {
			let na = if side == Side::Left { m } else { n };
			let (lda, ldb) = (na + 2, m + 1);
			let mut a = rng.mat_c64(na, na, lda);
			for j in 0..na {
				a[j * lda + j] = C64::new(na as f64 * 2.0 + 1.0 + j as f64, 0.5);
			}
			let b0 = rng.mat_c64(m, n, ldb);
			for transa in [Trans::Trans, Trans::ConjTrans] {
				// the explicit copy of op(A), at its own leading dimension
				let (oa, lda2) = op_c64(&a, lda, na, na, transa);
				for uplo in [Uplo::Upper, Uplo::Lower] {
					let flipped = if uplo == Uplo::Upper { Uplo::Lower } else { Uplo::Upper };
					for dg in [Diag::NonUnit, Diag::Unit] {
						for alpha in [C64::new(0.9, -0.3), C64::ONE] {
							let mut x1 = b0.clone();
							ztrsm(side, flipped, Trans::NoTrans, dg, m, n, alpha, &oa, lda2, &mut x1, ldb, &mut []).unwrap();
							let mut x2 = b0.clone();
							let mut w = vec![C64::new(f64::NAN, f64::NAN); ztrsm_copy_work(side, m, n)];
							ztrsm_copy(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x2, ldb, &mut w);
							let mut x3 = b0.clone();
							ztrsm_stream(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x3, ldb);
							let mut x4 = b0.clone();
							let mut w4 = vec![C64::new(f64::NAN, f64::NAN); ztrsm_work(side, transa, m, n)];
							ztrsm(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x4, ldb, &mut w4).unwrap();
							for j in 0..n {
								for i in 0..m {
									let x2v = x2[j * ldb + i];
									let x3v = x3[j * ldb + i];
									let x4v = x4[j * ldb + i];
									for (which, x2) in [("copy", &x2), ("stream", &x3), ("routed", &x4)] {
										assert!(
											bits_eq_c(x1[j * ldb + i], x2[j * ldb + i]),
											"ztrsm {which} vs explicit transpose {side:?} {uplo:?} {transa:?} {dg:?} m={m} n={n} ({i},{j})"
										);
									}
									let _ = (x2v, x3v, x4v);
								}
							}
						}
					}
				}
			}
		}
	}
}
