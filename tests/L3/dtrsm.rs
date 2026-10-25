use super::common::*;
use blas_wasm::L3::*;

#[test]
fn trsm_both_sides_residual() {
	let mut rng = Lcg(35);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				let mut a = rng.mat_f64(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = 2.0 * (n as f64) + 1.0 + j as f64;
				}
				let bcs = m + 1;
				let b0 = rng.mat_f64(m, n, bcs);
				let tri_at = |i: usize, j: usize| -> f64 {
					let in_tri = if upper { i <= j } else { i >= j };
					if !in_tri {
						0.0
					} else if unit && i == j {
						1.0
					} else {
						a[j * acs + i]
					}
				};
				let bound = 3.0 * (n as f64 + 1.0);

				// left: A·X = α·B0
				let mut x = b0.clone();
				dtrsm(Side::Left, uplo(upper), Trans::NoTrans, diag(unit), m, n, 0.9, &a, acs, &mut x, bcs, &mut vec![0.0f64; dtrsm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
				for j in 0..n {
					for i in 0..m {
						let ax = comp_sum((0..n).map(|l| tri_at(i, l) * x[j * bcs + l]));
						let want = 0.9 * b0[j * bcs + i];
						let xmax = (0..n).fold(0.0f64, |acc, l| acc.max(x[j * bcs + l].abs()));
						let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * bound * xmax + 1e-300;
						assert!(
							(ax - want).abs() <= tol,
							"dtrsm_left residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				// right: X·A = α·B0
				let mut x = b0.clone();
				dtrsm(Side::Right, uplo(upper), Trans::NoTrans, diag(unit), m, n, 0.9, &a, acs, &mut x, bcs, &mut vec![0.0f64; dtrsm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
				for j in 0..n {
					for i in 0..m {
						let xa = comp_sum((0..n).map(|l| x[l * bcs + i] * tri_at(l, j)));
						let want = 0.9 * b0[j * bcs + i];
						let xmax = (0..n).fold(0.0f64, |acc, l| acc.max(x[l * bcs + i].abs()));
						let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * bound * xmax + 1e-300;
						assert!(
							(xa - want).abs() <= tol,
							"dtrsm_right residual upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn trsm_bit_replay_both_sides() {
	let mut rng = Lcg(38);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- alpha * inv(A) * B, plain scal-then-dtrsv replay
				let acs = m + 1;
				let mut a = rng.mat_f64(m, m, acs);
				for j in 0..m {
					a[j * acs + j] = 2.0 * (m as f64) + 1.0 + j as f64;
				}
				let b0 = rng.mat_f64(m, n, bcs);
				let mut bt = b0.clone();
				dtrsm(Side::Left, uplo(upper), Trans::NoTrans, diag(unit), m, n, 0.9, &a, acs, &mut bt, bcs, &mut vec![0.0f64; dtrsm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
				let mut br = b0.clone();
				for j in 0..n {
					for i in 0..m {
						br[j * bcs + i] *= 0.9;
					}
					if upper {
						for l in (0..m).rev() {
							if !unit {
								br[j * bcs + l] /= a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
						}
					} else {
						for l in 0..m {
							if !unit {
								br[j * bcs + l] /= a[l * acs + l];
							}
							let t = -br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
						}
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "dtrsm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- alpha * B * inv(A)
				let acs = n + 1;
				let mut a = rng.mat_f64(n, n, acs);
				for j in 0..n {
					a[j * acs + j] = 2.0 * (n as f64) + 1.0 + j as f64;
				}
				let b0 = rng.mat_f64(m, n, bcs);
				let mut bt = b0.clone();
				dtrsm(Side::Right, uplo(upper), Trans::NoTrans, diag(unit), m, n, 0.9, &a, acs, &mut bt, bcs, &mut vec![0.0f64; dtrsm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
				let mut br = b0.clone();
				let elim = |br: &mut Vec<f64>, j: usize, k: usize| {
					let t = -a[j * acs + k];
					for i in 0..m {
						br[j * bcs + i] += br[k * bcs + i] * t;
					}
				};
				let finish = |br: &mut Vec<f64>, j: usize| {
					if !unit {
						let s = 1.0 / a[j * acs + j];
						for i in 0..m {
							br[j * bcs + i] *= s;
						}
					}
				};
				if upper {
					// plain ascending replay — the grouping keeps upper
					// fully ascending, so this asserts bit-identity to it
					for j in 0..n {
						for i in 0..m {
							br[j * bcs + i] *= 0.9;
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
								br[(gs + u) * bcs + i] *= 0.9;
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
							br[j * bcs + i] *= 0.9;
						}
						for k in j + 1..n {
							elim(&mut br, j, k);
						}
						finish(&mut br, j);
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "dtrsm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}

#[test]
fn dtrsm_transposed_candidates_equal_the_explicit_transpose_bitwise() {
	// the definition of the flag: op(A) through `transa` is bit-identical
	// to the no-transpose call on an explicitly transposed
	// copy with the triangle flipped. Both candidates (copy, stream) and
	// the routed wrapper, every side/uplo/diag, several shapes including
	// row counts off the kernels' block sizes, workspace NaN-filled.
	let mut rng = Lcg(0x7A11 + 5);
	for (m, n) in [(1, 1), (2, 3), (3, 2), (4, 4), (5, 7), (7, 5), (8, 9), (9, 8), (17, 13), (33, 6), (6, 33), (64, 65)] {
		for side in [Side::Left, Side::Right] {
			let na = if side == Side::Left { m } else { n };
			let (lda, ldb) = (na + 2, m + 1);
			let mut a = rng.mat_f64(na, na, lda);
			for j in 0..na {
				a[j * lda + j] = na as f64 * 2.0 + 1.0 + j as f64;
			}
			let b0 = rng.mat_f64(m, n, ldb);
			for transa in [Trans::Trans, Trans::ConjTrans] {
				// the explicit copy of op(A), at its own leading dimension
				let (oa, lda2) = op_f64(&a, lda, na, na, transa);
				for uplo in [Uplo::Upper, Uplo::Lower] {
					let flipped = if uplo == Uplo::Upper { Uplo::Lower } else { Uplo::Upper };
					for dg in [Diag::NonUnit, Diag::Unit] {
						for alpha in [0.9, 1.0] {
							let mut x1 = b0.clone();
							dtrsm(side, flipped, Trans::NoTrans, dg, m, n, alpha, &oa, lda2, &mut x1, ldb, &mut []).unwrap();
							let mut x2 = b0.clone();
							let mut w = vec![f64::NAN; dtrsm_copy_work(side, m, n)];
							dtrsm_copy(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x2, ldb, &mut w);
							let mut x3 = b0.clone();
							dtrsm_stream(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x3, ldb);
							let mut x4 = b0.clone();
							let mut w4 = vec![f64::NAN; dtrsm_work(side, transa, m, n)];
							dtrsm(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x4, ldb, &mut w4).unwrap();
							for j in 0..n {
								for i in 0..m {
									let x2v = x2[j * ldb + i];
									let x3v = x3[j * ldb + i];
									let x4v = x4[j * ldb + i];
									for (which, x2) in [("copy", &x2), ("stream", &x3), ("routed", &x4)] {
										assert!(
											x1[j * ldb + i].to_bits() == x2[j * ldb + i].to_bits(),
											"dtrsm {which} vs explicit transpose {side:?} {uplo:?} {transa:?} {dg:?} m={m} n={n} ({i},{j})"
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

#[test]
fn dtrsm_left_packed_bit_identical_to_the_sweep() {
	// the K1 candidate against the shipping left sweep: m across the
	// MR tails and the KC block edge, n across the 4-column tail, both
	// triangles, unit and non-unit, α ≠ 1; A's unstored triangle is
	// NaN-poisoned so a read of it would show
	let mut rng = Lcg(0x7A11 + 6);
	let shapes: Vec<(usize, usize)> = (1..14usize).map(|m| (m, 1 + (m * 7) % 11)).chain([(16, 4), (17, 9), (31, 12), (64, 64), (65, 33), (130, 20), (257, 8), (260, 12), (300, 5)]).collect();
	for &(m, n) in &shapes {
		for upper in [true, false] {
			for unit in [false, true] {
				let (acs, bcs) = (m + 1, m + 2);
				let mut a = rng.mat_f64(m, m, acs);
				for j in 0..m {
					for i in 0..m {
						if !(if upper { i <= j } else { i >= j }) {
							a[j * acs + i] = f64::NAN;
						}
					}
					a[j * acs + j] = 2.0 * m as f64 + 1.0; // diagonally dominant, as the solve tests use
				}
				let b0 = rng.mat_f64(m, n, bcs);
				for alpha in [1.0, -0.7] {
					let mut b1 = b0.clone();
					dtrsm(Side::Left, uplo(upper), Trans::NoTrans, diag(unit), m, n, alpha, &a, acs, &mut b1, bcs, &mut []).unwrap();
					let mut b2 = b0.clone();
					let mut work = vec![f64::NAN; dtrsm_left_packed_work(m, n)];
					dtrsm_left_packed(upper, unit, alpha, m, n, &a, acs, &mut b2, bcs, &mut work);
					for j in 0..n {
						for i in 0..m {
							assert_eq!(b1[j * bcs + i].to_bits(), b2[j * bcs + i].to_bits(), "dtrsm_left_packed vs sweep upper={upper} unit={unit} m={m} n={n} alpha={alpha} ({i},{j})");
						}
					}
				}
			}
		}
	}
}
