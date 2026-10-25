use super::common::*;
use blas_wasm::L3::*;

#[test]
fn syr2k_bit_for_bit() {
	let mut rng = Lcg(33);
	for &n in NS {
		let k = (n / 2).max(1);
		for upper in [true, false] {
			let (acs, bcs, ccs) = (n + 1, n + 2, n + 1);
			let a = rng.mat_f64(n, k, acs);
			let b = rng.mat_f64(n, k, bcs);
			let c0 = rng.mat_f64(n, n, ccs);


			let mut c = c0.clone();
			dsyr2k(uplo(upper), Trans::NoTrans, n, k, 0.5, &a, acs, &b, bcs, 0.25, &mut c, ccs, &mut vec![0.0f64; dsyr2k_work(Trans::NoTrans, n, k)]).unwrap();
			let mut cr = c0.clone();
			for j in 0..n {
				let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
				for i in lo..hi {
					cr[j * ccs + i] *= 0.25;
				}
				for l in 0..k {
					let tb = 0.5 * b[l * bcs + j];
					for i in lo..hi {
						cr[j * ccs + i] += a[l * acs + i] * tb;
					}
					let ta = 0.5 * a[l * acs + j];
					for i in lo..hi {
						cr[j * ccs + i] += b[l * bcs + i] * ta;
					}
				}
			}
			for j in 0..n {
				for i in 0..n {
					assert_eq!(c[j * ccs + i].to_bits(), cr[j * ccs + i].to_bits(), "dsyr2k ({i},{j})");
				}
			}
		}
	}
}

#[test]
fn dsyr2k_transposed_form_equals_the_explicit_transpose_bitwise() {
	// the definition of the flag: op(A) = Aᵀ through `trans` is bit-identical
	// to the no-transpose call on an explicitly transposed copy,
	// both triangles, with the workspace NaN-filled at the queried size
	let mut rng = Lcg(0x5EED_0000 + 6);
	for (n, k) in [(1, 1), (5, 3), (16, 16), (33, 70), (64, 63), (130, 9)] {
		for uplo in [Uplo::Upper, Uplo::Lower] {
			// stored shapes: A (and B) are k×n
			let (lda, ldb, ccs) = (k + 2, k + 1, n + 3);
			let a = rng.mat_f64(k, n, lda);
			let (oa, lda2) = op_f64(&a, lda, k, n, Trans::Trans);
			let b = rng.mat_f64(k, n, ldb);
			let (ob, ldb2) = op_f64(&b, ldb, k, n, Trans::Trans);
			let c0 = rng.mat_f64(n, n, ccs);
			for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4), (1.0, 1.0)] {
				let mut c1 = c0.clone();
				let mut w1 = vec![f64::NAN; dsyr2k_work(Trans::NoTrans, n, k)];
				dsyr2k(uplo, Trans::NoTrans, n, k, alpha, &oa, lda2, &ob, ldb2, beta, &mut c1, ccs, &mut w1).unwrap();
				let mut c2 = c0.clone();
				let mut w2 = vec![f64::NAN; dsyr2k_work(Trans::Trans, n, k)];
				dsyr2k(uplo, Trans::Trans, n, k, alpha, &a, lda, &b, ldb, beta, &mut c2, ccs, &mut w2).unwrap();
				for j in 0..n {
					for i in 0..n {
						assert!(
							c1[j * ccs + i].to_bits() == c2[j * ccs + i].to_bits(),
							"dsyr2k transposed vs explicit copy n={n} k={k} {uplo:?} ({i},{j})"
						);
					}
				}
			}
		}
	}
}
