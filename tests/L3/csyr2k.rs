use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::C32;

/// An n×k matrix at leading dimension `ld` as a tight n×k array.
fn cdense(a: &[C32], ld: usize, n: usize, k: usize) -> Vec<C32> {
	let mut d = vec![C32::ZERO; n * k];
	for l in 0..k {
		for i in 0..n {
			d[l * n + i] = a[l * ld + i];
		}
	}
	d
}

/// The explicit transpose of an n×k matrix at leading dimension `ld`,
/// k×n at leading dimension k + 1.
fn ctranspose(a: &[C32], ld: usize, n: usize, k: usize) -> Vec<C32> {
	let mut o = vec![C32::ZERO; (k + 1) * n.max(1)];
	for l in 0..k {
		for i in 0..n {
			o[i * (k + 1) + l] = a[l * ld + i];
		}
	}
	o
}

#[test]
fn csyr2k_bounded_triangle_only_and_transposed_form_bitwise() {
	// against a compensated reference on the stored triangle (the other
	// triangle must be untouched), then the transposed form against the
	// no-transpose call on an explicit transpose, bit for bit; ConjTrans
	// is an argument error for a symmetric update
	let mut rng = Lcg(0x5A2B + 6);
	for &(n, k) in &[(0usize, 3usize), (1, 1), (3, 2), (5, 7), (8, 8), (17, 5), (64, 9), (65, 130)] {
		for upper in [true, false] {
			let lda = n + 1;
			let a = rng.mat_c32(n, k, lda);
			let b = rng.mat_c32(n, k, lda);
			let ccs = n + 2;
			let c0 = rng.mat_c32(n, n, ccs);
			let alpha = C32::new(0.8, -0.3);
			let beta = C32::new(0.3, 0.5);
			let full_a = cdense(&a, lda, n, k);
			let full_b = cdense(&b, lda, n, k);
			let mut w = vec![C32::ZERO; csyr2k_work(Trans::NoTrans, n, k)];
			let mut c = c0.clone();
			csyr2k(uplo(upper), Trans::NoTrans, n, k, alpha, &a, lda, &b, lda, beta, &mut c, ccs, &mut w).unwrap();
			for j in 0..n {
				for i in 0..n {
					let stored = if upper { i <= j } else { i >= j };
					if !stored {
						assert!(bits_eq_cc(c[j * ccs + i], c0[j * ccs + i]), "csyr2k wrote outside the triangle ({i},{j})");
						continue;
					}
					let terms = |l: usize| full_a[l * n + i] * full_b[l * n + j] + full_b[l * n + i] * full_a[l * n + j];
					let want = c_up(alpha) * comp_sum_cc((0..k).map(terms)) + c_up(beta * c0[j * ccs + i]);
					let scale = comp_scale_cc((0..k).map(terms)) * (c_up(alpha).abs1() + 1.0) + c_up(beta * c0[j * ccs + i]).abs1();
					let tol = f32::EPSILON as f64 * (k.max(1) as f64) * 16.0 * scale + 1e-300;
					assert!((c_up(c[j * ccs + i]).re - want.re).abs() + (c_up(c[j * ccs + i]).im - want.im).abs() <= tol, "csyr2k upper={upper} n={n} k={k} ({i},{j})");
				}
			}
			let at = ctranspose(&a, lda, n, k);
			let bt = ctranspose(&b, lda, n, k);
			let mut w2 = vec![C32::new(f32::NAN, f32::NAN); csyr2k_work(Trans::Trans, n, k)];
			let mut c2 = c0.clone();
			csyr2k(uplo(upper), Trans::Trans, n, k, alpha, &at, k + 1, &bt, k + 1, beta, &mut c2, ccs, &mut w2).unwrap();
			for j in 0..n {
				for i in 0..n {
					assert!(bits_eq_cc(c[j * ccs + i], c2[j * ccs + i]), "csyr2k Trans vs explicit transpose ({i},{j})");
				}
			}
			assert_eq!(
				csyr2k(uplo(upper), Trans::ConjTrans, n, k, alpha, &at, k + 1, &bt, k + 1, beta, &mut c2, ccs, &mut w2),
				Err(blas_wasm::Error { routine: "csyr2k", arg: 2, kind: blas_wasm::ErrorKind::Argument })
			);
		}
	}
}
