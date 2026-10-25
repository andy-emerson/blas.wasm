use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::C64;

/// An n×k matrix at leading dimension `ld` as a tight n×k array.
fn zdense(a: &[C64], ld: usize, n: usize, k: usize) -> Vec<C64> {
	let mut d = vec![C64::ZERO; n * k];
	for l in 0..k {
		for i in 0..n {
			d[l * n + i] = a[l * ld + i];
		}
	}
	d
}

/// The explicit transpose of an n×k matrix at leading dimension `ld`,
/// k×n at leading dimension k + 1.
fn ztranspose(a: &[C64], ld: usize, n: usize, k: usize) -> Vec<C64> {
	let mut o = vec![C64::ZERO; (k + 1) * n.max(1)];
	for l in 0..k {
		for i in 0..n {
			o[i * (k + 1) + l] = a[l * ld + i];
		}
	}
	o
}

#[test]
fn zsyr2k_bounded_triangle_only_and_transposed_form_bitwise() {
	// against a compensated reference on the stored triangle (the other
	// triangle must be untouched), then the transposed form against the
	// no-transpose call on an explicit transpose, bit for bit; ConjTrans
	// is an argument error for a symmetric update
	let mut rng = Lcg(0x5A2B + 6);
	for &(n, k) in &[(0usize, 3usize), (1, 1), (3, 2), (5, 7), (8, 8), (17, 5), (64, 9), (65, 130)] {
		for upper in [true, false] {
			let lda = n + 1;
			let a = rng.mat_c64(n, k, lda);
			let b = rng.mat_c64(n, k, lda);
			let ccs = n + 2;
			let c0 = rng.mat_c64(n, n, ccs);
			let alpha = C64::new(0.8, -0.3);
			let beta = C64::new(0.3, 0.5);
			let full_a = zdense(&a, lda, n, k);
			let full_b = zdense(&b, lda, n, k);
			let mut w = vec![C64::ZERO; zsyr2k_work(Trans::NoTrans, n, k)];
			let mut c = c0.clone();
			zsyr2k(uplo(upper), Trans::NoTrans, n, k, alpha, &a, lda, &b, lda, beta, &mut c, ccs, &mut w).unwrap();
			for j in 0..n {
				for i in 0..n {
					let stored = if upper { i <= j } else { i >= j };
					if !stored {
						assert!(bits_eq_c(c[j * ccs + i], c0[j * ccs + i]), "zsyr2k wrote outside the triangle ({i},{j})");
						continue;
					}
					let terms = |l: usize| full_a[l * n + i] * full_b[l * n + j] + full_b[l * n + i] * full_a[l * n + j];
					let want = alpha * comp_sum_c((0..k).map(terms)) + beta * c0[j * ccs + i];
					let scale = comp_scale_c((0..k).map(terms)) * (alpha.abs1() + 1.0) + (beta * c0[j * ccs + i]).abs1();
					let tol = f64::EPSILON * (k.max(1) as f64) * 16.0 * scale + 1e-300;
					assert!((c[j * ccs + i].re - want.re).abs() + (c[j * ccs + i].im - want.im).abs() <= tol, "zsyr2k upper={upper} n={n} k={k} ({i},{j})");
				}
			}
			let at = ztranspose(&a, lda, n, k);
			let bt = ztranspose(&b, lda, n, k);
			let mut w2 = vec![C64::new(f64::NAN, f64::NAN); zsyr2k_work(Trans::Trans, n, k)];
			let mut c2 = c0.clone();
			zsyr2k(uplo(upper), Trans::Trans, n, k, alpha, &at, k + 1, &bt, k + 1, beta, &mut c2, ccs, &mut w2).unwrap();
			for j in 0..n {
				for i in 0..n {
					assert!(bits_eq_c(c[j * ccs + i], c2[j * ccs + i]), "zsyr2k Trans vs explicit transpose ({i},{j})");
				}
			}
			assert_eq!(
				zsyr2k(uplo(upper), Trans::ConjTrans, n, k, alpha, &at, k + 1, &bt, k + 1, beta, &mut c2, ccs, &mut w2),
				Err(blas_wasm::Error { routine: "zsyr2k", arg: 2, kind: blas_wasm::ErrorKind::Argument })
			);
		}
	}
}

#[test]
fn zsyr2k_replays_the_plain_sweep_bitwise() {
	// the dsyr2k shape's per-element sequence on complex lanes: βC on the
	// stored triangle, then per l ascending the B-column term
	// (α·B[j,l])·A[i,l] followed by the A-column term (α·A[j,l])·B[i,l]
	let mut rng = Lcg(0x5E2 + 3);
	let scale = |beta: C64, v: C64| if beta == C64::ZERO { C64::ZERO } else if beta == C64::ONE { v } else { beta * v };
	for n in 1..40usize {
		for upper in [true, false] {
			let k = 1 + (n * 5) % 11;
			let (lda, ldc) = (n + 1, n + 2);
			let a = rng.mat_c64(n, k, lda);
			let b = rng.mat_c64(n, k, lda);
			let c0 = rng.mat_c64(n, n, ldc);
			for (alpha, beta) in [(rng.next_c64(), rng.next_c64()), (C64::ONE, C64::ZERO), (rng.next_c64(), C64::ONE)] {
				let mut c = c0.clone();
				zsyr2k(uplo(upper), Trans::NoTrans, n, k, alpha, &a, lda, &b, lda, beta, &mut c, ldc, &mut []).unwrap();
				for j in 0..n {
					for i in 0..n {
						if !(if upper { i <= j } else { i >= j }) {
							assert!(bits_eq_c(c0[j * ldc + i], c[j * ldc + i]), "zsyr2k wrote outside the triangle ({i},{j})");
							continue;
						}
						let mut v = scale(beta, c0[j * ldc + i]);
						for l in 0..k {
							v = v + (alpha * b[l * lda + j]) * a[l * lda + i];
							v = v + (alpha * a[l * lda + j]) * b[l * lda + i];
						}
						assert!(bits_eq_c(v, c[j * ldc + i]), "zsyr2k replay upper={upper} n={n} k={k} ({i},{j})");
					}
				}
			}
		}
	}
}
