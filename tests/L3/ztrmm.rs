use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::C64;

#[test]
fn ztrmm_both_sides_bounded() {
	let mut rng = Lcg(135);
	let alpha = C64::new(0.9, -0.2);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			for unit in [true, false] {
				let acs = n + 1;
				let a = rng.mat_c64(n, n, acs);
				let bcs = m + 2;
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

				let mut b = b0.clone();
				ztrmm(Side::Left, uplo(upper), Trans::NoTrans, diag(unit), m, n, alpha, &a, acs, &mut b, bcs, &mut vec![C64::ZERO; ztrmm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
				for j in 0..n {
					for i in 0..m {
						let want =
							alpha * comp_sum_c((0..n).map(|l| tri_at(i, l) * b0[j * bcs + l]));
						let scale = comp_scale_c((0..n).map(|l| tri_at(i, l) * b0[j * bcs + l]))
							* (alpha.abs1() + 1.0);
						let tol = f64::EPSILON * (n.max(1) as f64) * 16.0 * scale + 1e-300;
						assert!(
							(b[j * bcs + i].re - want.re).abs()
								+ (b[j * bcs + i].im - want.im).abs()
								<= tol,
							"ztrmm_left upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}

				let mut b = b0.clone();
				ztrmm(Side::Right, uplo(upper), Trans::NoTrans, diag(unit), m, n, alpha, &a, acs, &mut b, bcs, &mut vec![C64::ZERO; ztrmm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
				for j in 0..n {
					for i in 0..m {
						let want =
							alpha * comp_sum_c((0..n).map(|l| b0[l * bcs + i] * tri_at(l, j)));
						let scale = comp_scale_c((0..n).map(|l| b0[l * bcs + i] * tri_at(l, j)))
							* (alpha.abs1() + 1.0);
						let tol = f64::EPSILON * (n.max(1) as f64) * 16.0 * scale + 1e-300;
						assert!(
							(b[j * bcs + i].re - want.re).abs()
								+ (b[j * bcs + i].im - want.im).abs()
								<= tol,
							"ztrmm_right upper={upper} unit={unit} n={n} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn ztrmm_bit_replay_both_sides() {
	let mut rng = Lcg(137);
	let alpha = C64::new(0.9, -0.2);
	for &(m, n) in TRI_DIMS {
		for upper in [true, false] {
			for unit in [true, false] {
				let bcs = m + 2;

				// left: B <- α·A·B, plain ztrmv-per-column replay then αscal
				let acs = m + 1;
				let a = rng.mat_c64(m, m, acs);
				let b0 = rng.mat_c64(m, n, bcs);
				let mut bt = b0.clone();
				ztrmm(Side::Left, uplo(upper), Trans::NoTrans, diag(unit), m, n, alpha, &a, acs, &mut bt, bcs, &mut vec![C64::ZERO; ztrmm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
				let mut br = b0.clone();
				for j in 0..n {
					if upper {
						for l in 0..m {
							let t = br[j * bcs + l];
							for i in 0..l {
								br[j * bcs + i] = br[j * bcs + i] + t * a[l * acs + i];
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					} else {
						for l in (0..m).rev() {
							let t = br[j * bcs + l];
							for i in l + 1..m {
								br[j * bcs + i] = br[j * bcs + i] + t * a[l * acs + i];
							}
							if !unit {
								br[j * bcs + l] = t * a[l * acs + l];
							}
						}
					}
					for i in 0..m {
						br[j * bcs + i] = alpha * br[j * bcs + i];
					}
				}
				for (x, y) in bt.iter().zip(&br) {
					if !x.re.is_nan() {
						assert!(bits_eq_c(*x, *y), "ztrmm_left {m}x{n} u={upper}");
					}
				}

				// right: B <- α·B·A
				let acs = n + 1;
				let a = rng.mat_c64(n, n, acs);
				let b0 = rng.mat_c64(m, n, bcs);
				let mut bt = b0.clone();
				ztrmm(Side::Right, uplo(upper), Trans::NoTrans, diag(unit), m, n, alpha, &a, acs, &mut bt, bcs, &mut vec![C64::ZERO; ztrmm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
				let mut br = b0.clone();
				let dcol = |br: &mut Vec<C64>, j: usize, lo: usize, hi: usize| {
					let d = if unit { C64::ONE } else { a[j * acs + j] };
					let s = alpha * d;
					for i in 0..m {
						br[j * bcs + i] = s * br[j * bcs + i];
					}
					for k in lo..hi {
						let t = alpha * a[j * acs + k];
						for i in 0..m {
							br[j * bcs + i] = br[j * bcs + i] + t * br[k * bcs + i];
						}
					}
				};
				if upper {
					// grouped replay: in-group (descending targets,
					// ascending k) first, then out-of-group sources
					let r = n % 4;
					let mut gs = n;
					while gs >= r + 4 {
						gs -= 4;
						for tc in (gs..gs + 4).rev() {
							dcol(&mut br, tc, gs, tc);
						}
						for k in 0..gs {
							for u in 0..4 {
								let t = alpha * a[(gs + u) * acs + k];
								for i in 0..m {
									br[(gs + u) * bcs + i] =
										br[(gs + u) * bcs + i] + t * br[k * bcs + i];
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
					if !x.re.is_nan() {
						assert!(bits_eq_c(*x, *y), "ztrmm_right {m}x{n} u={upper}");
					}
				}
			}
		}
	}
}

#[test]
fn ztrmm_transposed_candidates_equal_the_explicit_transpose_bitwise() {
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
			let a = rng.mat_c64(na, na, lda);
			let b0 = rng.mat_c64(m, n, ldb);
			for transa in [Trans::Trans, Trans::ConjTrans] {
				// the explicit copy of op(A), at its own leading dimension
				let (oa, lda2) = op_c64(&a, lda, na, na, transa);
				for uplo in [Uplo::Upper, Uplo::Lower] {
					let flipped = if uplo == Uplo::Upper { Uplo::Lower } else { Uplo::Upper };
					for dg in [Diag::NonUnit, Diag::Unit] {
						for alpha in [C64::new(0.9, -0.3), C64::ONE] {
							let mut x1 = b0.clone();
							ztrmm(side, flipped, Trans::NoTrans, dg, m, n, alpha, &oa, lda2, &mut x1, ldb, &mut []).unwrap();
							let mut x2 = b0.clone();
							let mut w = vec![C64::new(f64::NAN, f64::NAN); ztrmm_copy_work(side, m, n)];
							ztrmm_copy(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x2, ldb, &mut w);
							let mut x3 = b0.clone();
							ztrmm_stream(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x3, ldb);
							let mut x4 = b0.clone();
							let mut w4 = vec![C64::new(f64::NAN, f64::NAN); ztrmm_work(side, transa, m, n)];
							ztrmm(side, uplo, transa, dg, m, n, alpha, &a, lda, &mut x4, ldb, &mut w4).unwrap();
							for j in 0..n {
								for i in 0..m {
									let x2v = x2[j * ldb + i];
									let x3v = x3[j * ldb + i];
									let x4v = x4[j * ldb + i];
									for (which, x2) in [("copy", &x2), ("stream", &x3), ("routed", &x4)] {
										assert!(
											bits_eq_c(x1[j * ldb + i], x2[j * ldb + i]),
											"ztrmm {which} vs explicit transpose {side:?} {uplo:?} {transa:?} {dg:?} m={m} n={n} ({i},{j})"
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
