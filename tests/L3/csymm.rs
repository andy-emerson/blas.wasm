use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::C32;

/// The full symmetric matrix the stored triangle means, the other
/// triangle mirrored without conjugation, the diagonal an ordinary
/// complex element.
fn symmetric_pair(rng: &mut Lcg, n: usize, acs: usize, upper: bool) -> (Vec<C32>, Vec<C32>) {
	let mut full = vec![C32::ZERO; n * n];
	for j in 0..n {
		for i in 0..=j {
			let v = rng.next_c32();
			full[j * n + i] = v;
			full[i * n + j] = v;
		}
	}
	let nan = C32::new(f32::NAN, f32::NAN);
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
fn csymm_both_sides_bounded() {
	// against a compensated reference over the full symmetric matrix;
	// the unstored triangle is NaN-poisoned, so a read of it would show
	let mut rng = Lcg(0x5A11 + 1);
	for &n in NS {
		let m = n;
		for upper in [true, false] {
			let acs = n + 1;
			let (full, a) = symmetric_pair(&mut rng, n, acs, upper);
			let (bcs, ccs) = (m + 2, m + 1);
			let b = rng.mat_c32(m, n, bcs);
			let c0 = rng.mat_c32(m, n, ccs);
			let alpha = C32::new(0.8, -0.3);
			let beta = C32::new(0.3, 0.5);
			for side in [Side::Left, Side::Right] {
				let mut c = c0.clone();
				csymm(side, uplo(upper), m, n, alpha, &a, acs, &b, bcs, beta, &mut c, ccs).unwrap();
				for j in 0..n {
					for i in 0..m {
						let terms = |l: usize| if side == Side::Left { full[l * n + i] * b[j * bcs + l] } else { b[l * bcs + i] * full[j * n + l] };
						let want = c_up(alpha) * comp_sum_cc((0..n).map(terms)) + c_up(beta * c0[j * ccs + i]);
						let scale = comp_scale_cc((0..n).map(terms)) * (c_up(alpha).abs1() + 1.0) + c_up(beta * c0[j * ccs + i]).abs1();
						let tol = f32::EPSILON as f64 * (n.max(1) as f64) * 16.0 * scale + 1e-300;
						assert!((c_up(c[j * ccs + i]).re - want.re).abs() + (c_up(c[j * ccs + i]).im - want.im).abs() <= tol, "csymm {side:?} upper={upper} n={n} ({i},{j})");
					}
				}
			}
		}
	}
}

#[test]
fn csymm_right_replays_the_plain_sweep_bitwise() {
	// the right-side sweep's per-element sequence: βC, then l ascending
	// with the symmetric element looked up across the stored triangle and
	// the product (α·A[l,j])·B[i,l] added in that order — the header's claim
	let mut rng = Lcg(0x5A11 + 7);
	let scale = |beta: C32, v: C32| if beta == C32::ZERO { C32::ZERO } else if beta == C32::ONE { v } else { beta * v };
	for n in 1..40usize {
		for upper in [true, false] {
			let m = 1 + (n * 7) % 13;
			let (acs, bcs, ccs) = (n + 1, m + 1, m + 2);
			let (full, a) = symmetric_pair(&mut rng, n, acs, upper);
			let b = rng.mat_c32(m, n, bcs);
			let c0 = rng.mat_c32(m, n, ccs);
			for (alpha, beta) in [(rng.next_c32(), rng.next_c32()), (C32::ONE, C32::ZERO), (rng.next_c32(), C32::ONE)] {
				let mut c = c0.clone();
				csymm(Side::Right, uplo(upper), m, n, alpha, &a, acs, &b, bcs, beta, &mut c, ccs).unwrap();
				for j in 0..n {
					for i in 0..m {
						let mut v = scale(beta, c0[j * ccs + i]);
						for l in 0..n {
							v = v + (alpha * full[j * n + l]) * b[l * bcs + i];
						}
						assert!(bits_eq_cc(v, c[j * ccs + i]), "csymm right replay upper={upper} n={n} ({i},{j})");
					}
				}
			}
		}
	}
}
