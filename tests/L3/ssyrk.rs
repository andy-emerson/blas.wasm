use super::common::*;
use blas_wasm::L3::*;

#[test]
fn syrk_bit_for_bit() {
	let mut rng = Lcg(33);
	for &n in NS {
		let k = (n / 2).max(1);
		for upper in [true, false] {
			let (acs, ccs) = (n + 1, n + 1);
			let a = rng.mat_f32(n, k, acs);
			let c0 = rng.mat_f32(n, n, ccs);

			let mut c = c0.clone();
			ssyrk(uplo(upper), Trans::NoTrans, n, k, 0.5, &a, acs, 0.25, &mut c, ccs, &mut vec![0.0f32; ssyrk_work(Trans::NoTrans, n, k)]).unwrap();
			let mut cr = c0.clone();
			for j in 0..n {
				let (lo, hi) = if upper { (0, j + 1) } else { (j, n) };
				for i in lo..hi {
					cr[j * ccs + i] *= 0.25;
				}
				for l in 0..k {
					let t = 0.5 * a[l * acs + j];
					for i in lo..hi {
						cr[j * ccs + i] += a[l * acs + i] * t;
					}
				}
			}
			for j in 0..n {
				for i in 0..n {
					assert_eq!(c[j * ccs + i].to_bits(), cr[j * ccs + i].to_bits(), "ssyrk ({i},{j})");
				}
			}
		}
	}
}

#[test]
fn ssyrk_transposed_form_equals_the_explicit_transpose_bitwise() {
	// the definition of the flag: op(A) = Aᵀ through `trans` is bit-identical
	// to the no-transpose call on an explicitly transposed copy,
	// both triangles, with the workspace NaN-filled at the queried size
	let mut rng = Lcg(0x5EED_0000 + 5);
	for (n, k) in [(1, 1), (5, 3), (16, 16), (33, 70), (64, 63), (130, 9)] {
		for uplo in [Uplo::Upper, Uplo::Lower] {
			// stored shapes: A (and B) are k×n
			let (lda, ccs) = (k + 2, n + 3);
			let a = rng.mat_f32(k, n, lda);
			let (oa, lda2) = op_f32(&a, lda, k, n, Trans::Trans);

			let c0 = rng.mat_f32(n, n, ccs);
			for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4), (1.0, 1.0)] {
				let mut c1 = c0.clone();
				let mut w1 = vec![f32::NAN; ssyrk_work(Trans::NoTrans, n, k)];
				ssyrk(uplo, Trans::NoTrans, n, k, alpha, &oa, lda2, beta, &mut c1, ccs, &mut w1).unwrap();
				let mut c2 = c0.clone();
				let mut w2 = vec![f32::NAN; ssyrk_work(Trans::Trans, n, k)];
				ssyrk(uplo, Trans::Trans, n, k, alpha, &a, lda, beta, &mut c2, ccs, &mut w2).unwrap();
				for j in 0..n {
					for i in 0..n {
						assert!(
							c1[j * ccs + i].to_bits() == c2[j * ccs + i].to_bits(),
							"ssyrk transposed vs explicit copy n={n} k={k} {uplo:?} ({i},{j})"
						);
					}
				}
			}
		}
	}
}

#[test]
fn ssyrk_packed_bit_identical_to_the_sweep_and_never_writes_the_unstored_triangle() {
	// the K1 candidate against the shipping sweep: every n across the
	// MR/4 tails and the KC block edge, both triangles, three (α, β)
	// pairs; the unstored triangle is NaN-poisoned and must come back as
	// it went in
	let mut rng = Lcg(0x5E1 + 12);
	let sizes: Vec<(usize, usize)> = (0..24usize).map(|n| (n, 1 + (n * 5) % 9)).chain([(31, 40), (64, 64), (65, 257), (130, 300), (257, 12)]).collect();
	for &(n, k) in &sizes {
		for upper in [true, false] {
			let (acs, ccs) = (n + 1, n + 2);
			let a = rng.mat_f32(n, k, acs);
			let mut c0 = rng.mat_f32(n, n, ccs);
			for j in 0..n {
				for i in 0..n {
					if !(if upper { i <= j } else { i >= j }) {
						c0[j * ccs + i] = f32::NAN;
					}
				}
			}
			for (alpha, beta) in [(0.5, 0.25), (-1.3, 0.0), (0.7, 1.0)] {
				let mut c1 = c0.clone();
				ssyrk(uplo(upper), Trans::NoTrans, n, k, alpha, &a, acs, beta, &mut c1, ccs, &mut []).unwrap();
				let mut c2 = c0.clone();
				let mut work = vec![f32::NAN; ssyrk_packed_work(n, k)];
				ssyrk_packed(upper, alpha, n, k, &a, acs, beta, &mut c2, ccs, &mut work);
				for j in 0..n {
					for i in 0..n {
						let stored = if upper { i <= j } else { i >= j };
						if stored {
							assert_eq!(c1[j * ccs + i].to_bits(), c2[j * ccs + i].to_bits(), "ssyrk_packed vs sweep upper={upper} n={n} k={k} ({i},{j})");
						} else {
							assert!(c2[j * ccs + i].is_nan(), "ssyrk_packed wrote the unstored triangle upper={upper} n={n} ({i},{j})");
						}
					}
				}
			}
		}
	}
}
