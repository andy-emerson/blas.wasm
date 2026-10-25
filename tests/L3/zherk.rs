use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::C64;

#[test]
fn zherk_bit_for_bit_and_real_diagonal() {
	let mut rng = Lcg(133);
	let (alpha, beta) = (0.5f64, 0.25f64);
	for &n in NS {
		let k = (n / 2).max(1);
		for upper in [true, false] {
			let (acs, ccs) = (n + 1, n + 1);
			let a = rng.mat_c64(n, k, acs);
			let c0 = rng.mat_c64(n, n, ccs);

			let mut c = c0.clone();
			zherk(uplo(upper), Trans::NoTrans, n, k, alpha, &a, acs, beta, &mut c, ccs, &mut vec![C64::ZERO; zherk_work(Trans::NoTrans, n, k)]).unwrap();
			// replay: real-β component scale, l ascending with
			// t = α·conj(A[j,l]), then the diagonal's imag forced to 0
			let mut cr = c0.clone();
			for j in 0..n {
				let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
				for i in lo..hi {
					let v = cr[j * ccs + i];
					cr[j * ccs + i] = C64::new(v.re * beta, v.im * beta);
				}
				for l in 0..k {
					let t = a[l * acs + j].conj().scale(alpha);
					for i in lo..hi {
						cr[j * ccs + i] = cr[j * ccs + i] + t * a[l * acs + i];
					}
				}
				let d = cr[j * ccs + j];
				cr[j * ccs + j] = C64::new(d.re, 0.0);
			}
			for j in 0..n {
				for i in 0..n {
					assert!(
						bits_eq_c(c[j * ccs + i], cr[j * ccs + i]),
						"zherk upper={upper} n={n} ({i},{j})"
					);
				}
			}
			// Hermitian invariant: diagonal imag exactly +0.0
			for j in 0..n {
				assert_eq!(c[j * ccs + j].im.to_bits(), 0.0f64.to_bits(), "diag j={j}");
			}
		}
	}
}

#[test]
fn zherk_transposed_form_equals_the_explicit_transpose_bitwise() {
	// the definition of the flag: op(A) = Aᴴ through `trans` is bit-identical
	// to the no-transpose call on an explicitly conjugate-transposed copy,
	// both triangles, with the workspace NaN-filled at the queried size
	let mut rng = Lcg(0x5EED_0000 + 5);
	for (n, k) in [(1, 1), (5, 3), (16, 16), (33, 70), (64, 63), (130, 9)] {
		for uplo in [Uplo::Upper, Uplo::Lower] {
			// stored shapes: A (and B) are k×n
			let (lda, ccs) = (k + 2, n + 3);
			let a = rng.mat_c64(k, n, lda);
			let (oa, lda2) = op_c64(&a, lda, k, n, Trans::ConjTrans);

			let c0 = rng.mat_c64(n, n, ccs);
			for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4), (1.0, 1.0)] {
				let mut c1 = c0.clone();
				let mut w1 = vec![C64::new(f64::NAN, f64::NAN); zherk_work(Trans::NoTrans, n, k)];
				zherk(uplo, Trans::NoTrans, n, k, alpha, &oa, lda2, beta, &mut c1, ccs, &mut w1).unwrap();
				let mut c2 = c0.clone();
				let mut w2 = vec![C64::new(f64::NAN, f64::NAN); zherk_work(Trans::ConjTrans, n, k)];
				zherk(uplo, Trans::ConjTrans, n, k, alpha, &a, lda, beta, &mut c2, ccs, &mut w2).unwrap();
				for j in 0..n {
					for i in 0..n {
						assert!(
							bits_eq_c(c1[j * ccs + i], c2[j * ccs + i]),
							"zherk transposed vs explicit copy n={n} k={k} {uplo:?} ({i},{j})"
						);
					}
				}
			}
		}
	}
}
