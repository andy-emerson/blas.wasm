use super::common::*;
use blas_wasm::L3::*;

#[test]
fn gemm_bit_for_bit_and_bounded() {
	let mut rng = Lcg(31);
	for &(m, k, n) in DIMS {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_f32(m, k, acs);
		let b = rng.mat_f32(k, n, bcs);
		let c0 = rng.mat_f32(m, n, ccs);
		for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4)] {
			let mut c = c0.clone();
			sgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &a, acs, &b, bcs, beta, &mut c, ccs, &mut vec![0.0f32; sgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
			// same-order scalar replay (sgemv per column = column-saxpy)
			let mut cr = c0.clone();
			for j in 0..n {
				if beta == 0.0 {
					for i in 0..m {
						cr[j * ccs + i] = 0.0;
					}
				} else if beta != 1.0 {
					for i in 0..m {
						cr[j * ccs + i] *= beta;
					}
				}
				for l in 0..k {
					let t = alpha * b[j * bcs + l];
					for i in 0..m {
						cr[j * ccs + i] += a[l * acs + i] * t;
					}
				}
			}
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c[j * ccs + i].to_bits(),
						cr[j * ccs + i].to_bits(),
						"sgemm bits {m}x{k}x{n} ({i},{j})"
					);
				}
			}
			// independent bound, different accumulation order
			for j in 0..n {
				for i in 0..m {
					let want = alpha as f64 * comp_sum32((0..k).map(|l| a[l * acs + i] * b[j * bcs + l]))
						+ if beta == 0.0 { 0.0 } else { (beta * c0[j * ccs + i]) as f64 };
					let scale = comp_sum32((0..k).map(|l| (a[l * acs + i] * b[j * bcs + l]).abs()))
						+ c0[j * ccs + i].abs() as f64;
					let tol = f32::EPSILON as f64 * (k.max(1) as f64) * 8.0 * scale + 1e-40;
					assert!((c[j * ccs + i] as f64 - want).abs() <= tol, "sgemm bound ({i},{j})");
				}
			}
		}
	}
}

#[test]
fn gemm_tiled_bit_identical_to_gemm() {
	let mut rng = Lcg(36);
	// sizes crossing every tile boundary: exact multiples, tails in m,
	// n, both, and tiny
	for &(m, k, n) in &[(8usize, 8usize, 8usize), (4, 4, 4), (12, 7, 8), (9, 5, 10), (7, 3, 6), (3, 2, 3), (1, 1, 1), (0, 0, 0), (16, 16, 5), (5, 16, 16)] {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_f32(m, k, acs);
		let b = rng.mat_f32(k, n, bcs);
		let c0 = rng.mat_f32(m, n, ccs);
		for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4), (0.3, 1.0)] {
			let mut c1 = c0.clone();
			sgemm_colaxpy(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut cd = c0.clone();
			sgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &a, acs, &b, bcs, beta, &mut cd, ccs, &mut vec![0.0f32; sgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
			for j in 0..n {
				for i in 0..m {
					assert_eq!(c1[j * ccs + i].to_bits(), cd[j * ccs + i].to_bits(), "dispatcher vs colaxpy");
				}
			}
			let mut c2 = c0.clone();
			sgemm_tiled(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c2, ccs);
			let mut c3 = c0.clone();
			sgemm_col4(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c3, ccs);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c1[j * ccs + i].to_bits(),
						c2[j * ccs + i].to_bits(),
						"tiled vs column-saxpy {m}x{k}x{n} ({i},{j})"
					);
					assert_eq!(
						c1[j * ccs + i].to_bits(),
						c3[j * ccs + i].to_bits(),
						"col4 vs column-saxpy {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn sgemm_short_storage_is_an_error() {
	assert_eq!(sgemm(Trans::NoTrans, Trans::NoTrans, 2, 2, 2, 1.0, &[1.0; 4], 2, &[1.0; 3], 2, 0.0, &mut [0.0; 4], 2, &mut vec![0.0f32; sgemm_work(Trans::NoTrans, Trans::NoTrans, 2, 2, 2)]), Err(blas_wasm::Error { routine: "sgemm", arg: 9, kind: blas_wasm::ErrorKind::Length }));
}

#[test]
fn gemm_packed_bit_identical_to_colaxpy() {
	let mut rng = Lcg(42);
	// sizes crossing the packed-path boundaries: KC (256) exact / +1 /
	// with remainder, MC row-blocking, MR=8 row tails, 4-column tails
	for &(m, k, n) in &[
		(8usize, 8usize, 8usize),
		(12, 7, 8),
		(9, 5, 10),
		(1, 1, 1),
		(0, 0, 0),
		(5, 16, 16),
		(8, 300, 8),
		(140, 260, 12),
		(16, 256, 4),
		(33, 257, 7),
	] {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_f32(m, k, acs);
		let b = rng.mat_f32(k, n, bcs);
		let c0 = rng.mat_f32(m, n, ccs);
		for (alpha, beta) in [(1.0f32, 0.0f32), (-0.7, 0.4), (0.3, 1.0)] {
			let mut c1 = c0.clone();
			sgemm_colaxpy(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut c2 = c0.clone();
			// a NaN-filled workspace of exactly the queried size: any read of
			// scratch the packer did not write would poison C
			let mut work = vec![f32::NAN; sgemm_packed_work(m, n, k)];
			sgemm_packed(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c2, ccs, &mut work);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c1[j * ccs + i].to_bits(),
						c2[j * ccs + i].to_bits(),
						"packed vs column-saxpy {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn gemm_dispatch_packed_zone_bit_identical() {
	// A = 1032x800x4B = 3.3 MB > the 3 MB tiled threshold — the
	// dispatcher routes packed; replay against colaxpy
	let mut rng = Lcg(44);
	let (m, k, n) = (1032, 800, 8);
	let (acs, bcs, ccs) = (m, k, m);
	let a = rng.mat_f32(m, k, acs);
	let b = rng.mat_f32(k, n, bcs);
	let c0 = rng.mat_f32(m, n, ccs);
	let mut c1 = c0.clone();
	sgemm_colaxpy(-0.7, m, n, k, &a, acs, &b, bcs, 0.4, &mut c1, ccs);
	let mut c2 = c0.clone();
	sgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, -0.7, &a, acs, &b, bcs, 0.4, &mut c2, ccs, &mut vec![0.0f32; sgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
	for i in 0..c1.len() {
		assert_eq!(c1[i].to_bits(), c2[i].to_bits(), "dispatch(packed zone) vs colaxpy @{i}");
	}
}

#[test]
fn sgemm_flags_equal_the_explicit_transpose_bitwise() {
	// the definition of the flags: op(A)·op(B) through transa/transb is
	// bit-identical to the no-transpose call on explicitly transposed
	// (and conjugated) copies, for every flag pair, below and above the
	// packed threshold, with the workspace NaN-filled at the queried size
	let mut rng = Lcg(0x5EED_5EED);
	let flags = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
	for (m, n, k) in [(1, 1, 1), (5, 7, 3), (16, 16, 16), (33, 9, 70), (64, 65, 63), (900, 12, 900)] {
		for ta in flags {
			for tb in flags {
				// stored shapes: A is m×k or k×m, B is k×n or n×k
				let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
				let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
				let (lda, ldb, ccs) = (ar + 2, br + 1, m + 3);
				let a = rng.mat_f32(ar, ac, lda);
				let b = rng.mat_f32(br, bc, ldb);
				let c0 = rng.mat_f32(m, n, ccs);
				let (oa, lda2) = op_f32(&a, lda, ar, ac, ta);
				let (ob, ldb2) = op_f32(&b, ldb, br, bc, tb);
				for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4), (1.0, 1.0)] {
					let mut c1 = c0.clone();
					let mut w1 = vec![f32::NAN; sgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)];
					sgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &oa, lda2, &ob, ldb2, beta, &mut c1, ccs, &mut w1).unwrap();
					let mut c2 = c0.clone();
					let mut w2 = vec![f32::NAN; sgemm_work(ta, tb, m, n, k)];
					sgemm(ta, tb, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c2, ccs, &mut w2).unwrap();
					for j in 0..n {
						for i in 0..m {
							assert_eq!(
								c1[j * ccs + i].to_bits(),
								c2[j * ccs + i].to_bits(),
								"sgemm({ta:?},{tb:?}) vs explicit copy {m}x{k}x{n} ({i},{j})"
							);
						}
					}
				}
			}
		}
	}
}

#[test]
fn sgemm_packed_op_equals_the_copy_candidate_bitwise() {
	// the second D3 candidate: packing from a transposed read, with the
	// gathered tails and the padded column group, is bit-identical to the
	// routed wrapper (the whole-operand copy) for every flag pair on
	// shapes that exercise every tail: rows off MR, columns off 4, k over
	// KC, m below MR, n below 4; workspace NaN-filled at the queried size
	let mut rng = Lcg(0xBEEF + 1);
	let flags = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
	for (m, n, k) in [(1, 1, 1), (3, 3, 5), (5, 7, 3), (7, 2, 300), (9, 5, 257), (13, 11, 16), (130, 9, 300), (33, 4, 70), (2, 9, 20), (16, 16, 16)] {
		for ta in flags {
			for tb in flags {
				let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
				let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
				let (lda, ldb, ccs) = (ar + 1, br + 2, m + 1);
				let a = rng.mat_f32(ar, ac, lda);
				let b = rng.mat_f32(br, bc, ldb);
				let c0 = rng.mat_f32(m, n, ccs);
				for (alpha, beta) in [(1.0, 0.0), (-0.7, 0.4), (1.0, 1.0)] {
					let mut c1 = c0.clone();
					let mut w1 = vec![f32::NAN; sgemm_work(ta, tb, m, n, k)];
					sgemm(ta, tb, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c1, ccs, &mut w1).unwrap();
					// the routed wrapper may take the tiled shape below the threshold; the
					// packed shape is bit-identical to it, so the comparison still holds
					let mut c2 = c0.clone();
					let mut w2 = vec![f32::NAN; sgemm_packed_op_work(ta, tb, m, n, k)];
					sgemm_packed_op(ta, tb, alpha, m, n, k, &a, lda, &b, ldb, beta, &mut c2, ccs, &mut w2);
					for j in 0..n {
						for i in 0..m {
							assert!(
								c1[j * ccs + i].to_bits() == c2[j * ccs + i].to_bits(),
								"sgemm_packed_op({ta:?},{tb:?}) vs copy {m}x{k}x{n} ({i},{j})"
							);
						}
					}
				}
			}
		}
	}
	assert_eq!(sgemm_packed_op_work(Trans::NoTrans, Trans::NoTrans, 300, 40, 300), sgemm_packed_work(300, 40, 300));
}
