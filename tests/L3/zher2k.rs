use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::C64;

#[test]
fn zher2k_bit_for_bit_and_real_diagonal() {
	let mut rng = Lcg(134);
	let alpha = C64::new(0.5, -0.7);
	let beta = 0.25f64;
	for &n in NS {
		let k = (n / 2).max(1);
		for upper in [true, false] {
			let (acs, bcs, ccs) = (n + 1, n + 2, n + 1);
			let a = rng.mat_c64(n, k, acs);
			let b = rng.mat_c64(n, k, bcs);
			let c0 = rng.mat_c64(n, n, ccs);

			let mut c = c0.clone();
			zher2k(uplo(upper), Trans::NoTrans, n, k, alpha, &a, acs, &b, bcs, beta, &mut c, ccs, &mut vec![C64::ZERO; zher2k_work(Trans::NoTrans, n, k)]).unwrap();
			// replay: real-β component scale; per l the A-sourced adds
			// (t = α·conj(B[j,l])) precede the B-sourced adds
			// (t = conj(α·A[j,l])); diagonal imag forced to 0 at the end
			let mut cr = c0.clone();
			for j in 0..n {
				let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
				for i in lo..hi {
					let v = cr[j * ccs + i];
					cr[j * ccs + i] = C64::new(v.re * beta, v.im * beta);
				}
				for l in 0..k {
					let tb = alpha * b[l * bcs + j].conj();
					for i in lo..hi {
						cr[j * ccs + i] = cr[j * ccs + i] + tb * a[l * acs + i];
					}
					let ta = (alpha * a[l * acs + j]).conj();
					for i in lo..hi {
						cr[j * ccs + i] = cr[j * ccs + i] + ta * b[l * bcs + i];
					}
				}
				let d = cr[j * ccs + j];
				cr[j * ccs + j] = C64::new(d.re, 0.0);
			}
			for j in 0..n {
				for i in 0..n {
					assert!(
						bits_eq_c(c[j * ccs + i], cr[j * ccs + i]),
						"zher2k upper={upper} n={n} ({i},{j})"
					);
				}
			}
			for j in 0..n {
				assert_eq!(c[j * ccs + j].im.to_bits(), 0.0f64.to_bits(), "diag j={j}");
			}
		}
	}
}

#[test]
fn zher2k_transposed_form_equals_the_explicit_transpose_bitwise() {
	// the definition of the flag: op(A) = Aᴴ through `trans` is bit-identical
	// to the no-transpose call on an explicitly conjugate-transposed copy,
	// both triangles, with the workspace NaN-filled at the queried size
	let mut rng = Lcg(0x5EED_0000 + 6);
	for (n, k) in [(1, 1), (5, 3), (16, 16), (33, 70), (64, 63), (130, 9)] {
		for uplo in [Uplo::Upper, Uplo::Lower] {
			// stored shapes: A (and B) are k×n
			let (lda, ldb, ccs) = (k + 2, k + 1, n + 3);
			let a = rng.mat_c64(k, n, lda);
			let (oa, lda2) = op_c64(&a, lda, k, n, Trans::ConjTrans);
			let b = rng.mat_c64(k, n, ldb);
			let (ob, ldb2) = op_c64(&b, ldb, k, n, Trans::ConjTrans);
			let c0 = rng.mat_c64(n, n, ccs);
			for (alpha, beta) in [(C64::ONE, 0.0), (C64::new(-0.7, 0.2), 0.4), (C64::ONE, 1.0)] {
				let mut c1 = c0.clone();
				let mut w1 = vec![C64::new(f64::NAN, f64::NAN); zher2k_work(Trans::NoTrans, n, k)];
				zher2k(uplo, Trans::NoTrans, n, k, alpha, &oa, lda2, &ob, ldb2, beta, &mut c1, ccs, &mut w1).unwrap();
				let mut c2 = c0.clone();
				let mut w2 = vec![C64::new(f64::NAN, f64::NAN); zher2k_work(Trans::ConjTrans, n, k)];
				zher2k(uplo, Trans::ConjTrans, n, k, alpha, &a, lda, &b, ldb, beta, &mut c2, ccs, &mut w2).unwrap();
				for j in 0..n {
					for i in 0..n {
						assert!(
							bits_eq_c(c1[j * ccs + i], c2[j * ccs + i]),
							"zher2k transposed vs explicit copy n={n} k={k} {uplo:?} ({i},{j})"
						);
					}
				}
			}
		}
	}
}
