use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::C64;

/// The full symmetric matrix the stored triangle means, the other
/// triangle mirrored without conjugation, the diagonal an ordinary
/// complex element.
fn symmetric_pair(rng: &mut Lcg, n: usize, acs: usize, upper: bool) -> (Vec<C64>, Vec<C64>) {
	let mut full = vec![C64::ZERO; n * n];
	for j in 0..n {
		for i in 0..=j {
			let v = rng.next_c64();
			full[j * n + i] = v;
			full[i * n + j] = v;
		}
	}
	let nan = C64::new(f64::NAN, f64::NAN);
	let mut a = vec![nan; if n == 0 { 0 } else { acs * (n - 1) + n }];
	for j in 0..n {
		for i in 0..n {
			if if upper { i <= j } else { i >= j } {
				a[j * acs + i] = full[j * n + i];
			}
		}
	}
	(full, a)
}

#[test]
fn zsymm_both_sides_bounded() {
	// against a compensated reference over the full symmetric matrix;
	// the unstored triangle is NaN-poisoned, so a read of it would show
	let mut rng = Lcg(0x5A11 + 1);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			let acs = n + 1;
			let (full, a) = symmetric_pair(&mut rng, n, acs, upper);
			let (bcs, ccs) = (m + 2, m + 1);
			let b = rng.mat_c64(m, n, bcs);
			let c0 = rng.mat_c64(m, n, ccs);
			let alpha = C64::new(0.8, -0.3);
			let beta = C64::new(0.3, 0.5);
			for side in [Side::Left, Side::Right] {
				let mut c = c0.clone();
				zsymm(side, uplo(upper), m, n, alpha, &a, acs, &b, bcs, beta, &mut c, ccs).unwrap();
				for j in 0..n {
					for i in 0..m {
						let terms = |l: usize| if side == Side::Left { full[l * n + i] * b[j * bcs + l] } else { b[l * bcs + i] * full[j * n + l] };
						let want = alpha * comp_sum_c((0..n).map(terms)) + beta * c0[j * ccs + i];
						let scale = comp_scale_c((0..n).map(terms)) * (alpha.abs1() + 1.0) + (beta * c0[j * ccs + i]).abs1();
						let tol = f64::EPSILON * (n.max(1) as f64) * 16.0 * scale + 1e-300;
						assert!((c[j * ccs + i].re - want.re).abs() + (c[j * ccs + i].im - want.im).abs() <= tol, "zsymm {side:?} upper={upper} n={n} ({i},{j})");
					}
				}
			}
		}
	}
}

#[test]
fn zsymm_right_replays_the_plain_sweep_bitwise() {
	// the right-side sweep's per-element sequence: βC, then l ascending
	// with the symmetric element looked up across the stored triangle and
	// the product (α·A[l,j])·B[i,l] added in that order — the header's claim
	let mut rng = Lcg(0x5A11 + 7);
	let scale = |beta: C64, v: C64| if beta == C64::ZERO { C64::ZERO } else if beta == C64::ONE { v } else { beta * v };
	for n in 1..40usize {
		for upper in [true, false] {
			let m = 1 + (n * 7) % 13;
			let (acs, bcs, ccs) = (n + 1, m + 1, m + 2);
			let (full, a) = symmetric_pair(&mut rng, n, acs, upper);
			let b = rng.mat_c64(m, n, bcs);
			let c0 = rng.mat_c64(m, n, ccs);
			for (alpha, beta) in [(rng.next_c64(), rng.next_c64()), (C64::ONE, C64::ZERO), (rng.next_c64(), C64::ONE)] {
				let mut c = c0.clone();
				zsymm(Side::Right, uplo(upper), m, n, alpha, &a, acs, &b, bcs, beta, &mut c, ccs).unwrap();
				for j in 0..n {
					for i in 0..m {
						let mut v = scale(beta, c0[j * ccs + i]);
						for l in 0..n {
							v = v + (alpha * full[j * n + l]) * b[l * bcs + i];
						}
						assert!(bits_eq_c(v, c[j * ccs + i]), "zsymm right replay upper={upper} n={n} ({i},{j})");
					}
				}
			}
		}
	}
}
