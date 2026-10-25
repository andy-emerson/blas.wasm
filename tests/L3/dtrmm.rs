use super::common::*;
use blas_wasm::L3::*;

#[test]
fn trmm_both_sides_bounded() {
	let mut rng = Lcg(34);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				let a = rng.mat_f64(n, n, acs);
				let bcs = m + 2;
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

				let mut b = b0.clone();
				dtrmm(Side::Left, uplo(upper), Trans::NoTrans, diag(unit), m, n, 0.9, &a, acs, &mut b, bcs, &mut vec![0.0f64; dtrmm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
				for j in 0..n {
					for i in 0..m {
						let want = 0.9 * comp_sum((0..n).map(|l| tri_at(i, l) * b0[j * bcs + l]));
						let scale = comp_sum((0..n).map(|l| (tri_at(i, l) * b0[j * bcs + l]).abs()));
						let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * scale + 1e-300;
						assert!(
							(b[j * bcs + i] - want).abs() <= tol,
							"dtrmm_left upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				let mut b = b0.clone();
				dtrmm(Side::Right, uplo(upper), Trans::NoTrans, diag(unit), m, n, 0.9, &a, acs, &mut b, bcs, &mut vec![0.0f64; dtrmm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
				for j in 0..n {
					for i in 0..m {
						let want = 0.9 * comp_sum((0..n).map(|l| b0[l * bcs + i] * tri_at(l, j)));
						let scale = comp_sum((0..n).map(|l| (b0[l * bcs + i] * tri_at(l, j)).abs()));
						let tol = f64::EPSILON * (n.max(1) as f64) * 8.0 * scale + 1e-300;
						assert!(
							(b[j * bcs + i] - want).abs() <= tol,
							"dtrmm_right upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn trmm_bit_replay_both_sides() {
	let mut rng = Lcg(37);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- alpha * A * B, plain dtrmv-per-column replay
				let acs = m + 1;
				let a = rng.mat_f64(m, m, acs);
				let b0 = rng.mat_f64(m, n, bcs);
				let mut bt = b0.clone();
				dtrmm(Side::Left, uplo(upper), Trans::NoTrans, diag(unit), m, n, 0.9, &a, acs, &mut bt, bcs, &mut vec![0.0f64; dtrmm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
				let mut br = b0.clone();
				for j in 0..n {
					if upper {
						for l in 0..m {
							let t = br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					} else {
						for l in (0..m).rev() {
							let t = br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] += a[l * acs + i] * t;
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					}
					for i in 0..m {
						br[j * bcs + i] *= 0.9;
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "dtrmm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- alpha * B * A
				let acs = n + 1;
				let a = rng.mat_f64(n, n, acs);
				let b0 = rng.mat_f64(m, n, bcs);
				let mut bt = b0.clone();
				dtrmm(Side::Right, uplo(upper), Trans::NoTrans, diag(unit), m, n, 0.9, &a, acs, &mut bt, bcs, &mut vec![0.0f64; dtrmm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
				let mut br = b0.clone();
				let dcol = |br: &mut Vec<f64>, j: usize, lo: usize, hi: usize| {
					let d = if unit { 1.0 } else { a[j * acs + j] };
					let s = 0.9 * d;
					for i in 0..m {
						br[j * bcs + i] *= s;
					}
					for k in lo..hi {
						let t = 0.9 * a[j * acs + k];
						for i in 0..m {
							br[j * bcs + i] += br[k * bcs + i] * t;
						}
					}
				};
				if upper {
					// grouped replay: in-group (descending targets, ascending
					// k) first, then out-of-group sources ascending
					let r = n % 4;
					let mut gs = n;
					while gs >= r + 4 {
						gs -= 4;
						for tc in (gs..gs + 4).rev() {
							dcol(&mut br, tc, gs, tc);
						}
						for k in 0..gs {
							for u in 0..4 {
								let t = 0.9 * a[(gs + u) * acs + k];
								for i in 0..m {
									br[(gs + u) * bcs + i] += br[k * bcs + i] * t;
								}
							}
						}
					}
					for j in (0..r).rev() {
						dcol(&mut br, j, 0, j);
					}
				} else {
					// plain ascending replay — the grouping keeps lower
					// fully ascending, so this asserts bit-identity to it
					for j in 0..n {
						dcol(&mut br, j, j + 1, n);
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.is_nan() {
						assert_eq!(x.to_bits(), y.to_bits(), "dtrmm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}

#[test]
fn dtrmm_transposed_candidates_equal_the_explicit_transpose_bitwise() {
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
			let a = rng.mat_f64(na, na, lda);
			let b0 = rng.mat_f64(m, n, ldb);
			for transa in [Trans::Trans, Trans::ConjTrans] {
				// the explicit copy of op(A), at its own leading dimension
				let (oa, lda2) = op_f64(&a, lda, na, na, transa);
				for uplo in [Uplo::Upper, Uplo::Lower] {
					let flipped = if uplo == Uplo::Upper { Uplo::Lower } else { Uplo::Upper };
					for dg in [Diag::NonUnit, Diag::Unit] {
						for alpha in [0.9, 1.0] {
							let mut x1 = b0.clone();
							dtrmm(side, flipped, Trans::NoTrans, dg, m, n, alpha, &oa, lda2, &mut x1, ldb, &mut []).unwrap();
							let mut x2 = b0.clone();
							let mut w = vec![f64::NAN; dtrmm_copy_work(side, m, n)];
							dtrmm_copy(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x2, ldb, &mut w);
							let mut x3 = b0.clone();
							dtrmm_stream(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x3, ldb);
							let mut x4 = b0.clone();
							let mut w4 = vec![f64::NAN; dtrmm_work(side, transa, m, n)];
							dtrmm(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x4, ldb, &mut w4).unwrap();
							for j in 0..n {
								for i in 0..m {
									let x2v = x2[j * ldb + i];
									let x3v = x3[j * ldb + i];
									let x4v = x4[j * ldb + i];
									for (which, x2) in [("copy", &x2), ("stream", &x3), ("routed", &x4)] {
										assert!(
											x1[j * ldb + i].to_bits() == x2[j * ldb + i].to_bits(),
											"dtrmm {which} vs explicit transpose {side:?} {uplo:?} {transa:?} {dg:?} m={m} n={n} ({i},{j})"
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
