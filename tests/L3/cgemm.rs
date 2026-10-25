use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::C64;
use blas_wasm::C32;

#[test]
fn cgemm_bit_for_bit_and_bounded() {
	let mut rng = Lcg(131);
	for &(m, k, n) in DIMS {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_c32(m, k, acs);
		let b = rng.mat_c32(k, n, bcs);
		let c0 = rng.mat_c32(m, n, ccs);
		for (alpha, beta) in [
			(C32::ONE, C32::ZERO),
			(C32::new(-0.7, 0.4), C32::new(0.4, -0.2)),
		] {
			let mut c = c0.clone();
			cgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &a, acs, &b, bcs, beta, &mut c, ccs, &mut vec![C32::ZERO; cgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
			// same-order scalar replay (cgemv per column = column-caxpy)
			let mut cr = c0.clone();
			for j in 0..n {
				if beta == C32::ZERO {
					for i in 0..m {
						cr[j * ccs + i] = C32::ZERO;
					}
				} else if beta != C32::ONE {
					for i in 0..m {
						cr[j * ccs + i] = beta * cr[j * ccs + i];
					}
				}
				for l in 0..k {
					let t = alpha * b[j * bcs + l];
					for i in 0..m {
						cr[j * ccs + i] = cr[j * ccs + i] + t * a[l * acs + i];
					}
				}
			}
			for j in 0..n {
				for i in 0..m {
					assert!(
						bits_eq_cc(c[j * ccs + i], cr[j * ccs + i]),
						"cgemm bits {m}x{k}x{n} ({i},{j})"
					);
				}
			}
			// independent bound, different accumulation order
			for j in 0..n {
				for i in 0..m {
					let bt = if beta == C32::ZERO { C64::ZERO } else { c_up(beta * c0[j * ccs + i]) };
					let want =
						c_up(alpha) * comp_sum_cc((0..k).map(|l| a[l * acs + i] * b[j * bcs + l])) + bt;
					let scale = comp_scale_cc((0..k).map(|l| a[l * acs + i] * b[j * bcs + l]))
						* (alpha.abs1() as f64 + 1.0)
						+ bt.abs1();
					let tol = EPS * (k.max(1) as f64) * 16.0 * scale + 1e-30;
					assert!(
						(c[j * ccs + i].re as f64 - want.re).abs() + (c[j * ccs + i].im as f64 - want.im).abs()
							<= tol,
						"cgemm bound ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn cgemm_col4_bit_identical_to_colaxpy() {
	let mut rng = Lcg(136);
	// sizes crossing the 4-column group boundary: multiples, tails, tiny
	for &(m, k, n) in &[
		(8usize, 8usize, 8usize),
		(4, 4, 4),
		(12, 7, 8),
		(9, 5, 10),
		(7, 3, 6),
		(3, 2, 3),
		(1, 1, 1),
		(0, 0, 0),
		(16, 16, 5),
		(5, 16, 16),
	] {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_c32(m, k, acs);
		let b = rng.mat_c32(k, n, bcs);
		let c0 = rng.mat_c32(m, n, ccs);
		for (alpha, beta) in [
			(C32::ONE, C32::ZERO),
			(C32::new(-0.7, 0.2), C32::new(0.4, 0.1)),
			(C32::new(0.3, -0.6), C32::ONE),
		] {
			let mut c1 = c0.clone();
			cgemm_colaxpy(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut c2 = c0.clone();
			cgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &a, acs, &b, bcs, beta, &mut c2, ccs, &mut vec![C32::ZERO; cgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
			for j in 0..n {
				for i in 0..m {
					assert!(
						bits_eq_cc(c1[j * ccs + i], c2[j * ccs + i]),
						"cgemm col4 vs colaxpy {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn cgemm_short_storage_is_an_error() {
	let a = [C32::ONE; 4];
	let b = [C32::ONE; 3];
	assert_eq!(cgemm(Trans::NoTrans, Trans::NoTrans, 2, 2, 2, C32::ONE, &a, 2, &b, 2, C32::ZERO, &mut [C32::ZERO; 4], 2, &mut vec![C32::ZERO; cgemm_work(Trans::NoTrans, Trans::NoTrans, 2, 2, 2)]), Err(blas_wasm::Error { routine: "cgemm", arg: 9, kind: blas_wasm::ErrorKind::Length }));
}

#[test]
fn cgemm_packed_bit_identical_to_colaxpy() {
	let mut rng = Lcg(142);
	// sizes crossing the packed-path boundaries: KC (256) exact / +1 /
	// with remainder, MC row-blocking, MR=4 row tails, column tails
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
		let a = rng.mat_c32(m, k, acs);
		let b = rng.mat_c32(k, n, bcs);
		let c0 = rng.mat_c32(m, n, ccs);
		for (alpha, beta) in [
			(C32::ONE, C32::ZERO),
			(C32::new(-0.7, 0.2), C32::new(0.4, 0.1)),
			(C32::new(0.3, -0.6), C32::ONE),
		] {
			let mut c1 = c0.clone();
			cgemm_colaxpy(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut c2 = c0.clone();
			// a NaN-filled workspace of exactly the queried size: any read of
			// scratch the packer did not write would poison C
			let mut work = vec![C32::new(f32::NAN, f32::NAN); cgemm_packed_work(m, n, k)];
			cgemm_packed(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c2, ccs, &mut work);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c1[j * ccs + i].re.to_bits(),
						c2[j * ccs + i].re.to_bits(),
						"packed vs column-caxpy re {m}x{k}x{n} ({i},{j})"
					);
					assert_eq!(
						c1[j * ccs + i].im.to_bits(),
						c2[j * ccs + i].im.to_bits(),
						"packed vs column-caxpy im {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn cgemm_dispatch_packed_zone_bit_identical() {
	// A = 1040x1024x8B = 8.1 MB >= the 8 MB packed threshold — the
	// dispatcher routes packed; replay against colaxpy
	let mut rng = Lcg(144);
	let (m, k, n) = (1040, 1024, 8);
	let (acs, bcs, ccs) = (m, k, m);
	let a = rng.mat_c32(m, k, acs);
	let b = rng.mat_c32(k, n, bcs);
	let c0 = rng.mat_c32(m, n, ccs);
	let (alpha, beta) = (C32::new(-0.7, 0.2), C32::new(0.4, 0.1));
	let mut c1 = c0.clone();
	cgemm_colaxpy(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c1, ccs);
	let mut c2 = c0.clone();
	cgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &a, acs, &b, bcs, beta, &mut c2, ccs, &mut vec![C32::ZERO; cgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
	for i in 0..c1.len() {
		assert_eq!(c1[i].re.to_bits(), c2[i].re.to_bits(), "dispatch(packed) re @{i}");
		assert_eq!(c1[i].im.to_bits(), c2[i].im.to_bits(), "dispatch(packed) im @{i}");
	}
}

#[test]
fn cgemm_flags_equal_the_explicit_transpose_bitwise() {
	// the definition of the flags: op(A)·op(B) through transa/transb is
	// bit-identical to the no-transpose call on explicitly transposed
	// (and conjugated) copies, for every flag pair, below and above the
	// packed threshold, with the workspace NaN-filled at the queried size
	let mut rng = Lcg(0x5EED_5EED);
	let flags = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
	for (m, n, k) in [(1, 1, 1), (5, 7, 3), (16, 16, 16), (33, 9, 70), (64, 65, 63), (1024, 8, 1100)] {
		for ta in flags {
			for tb in flags {
				// stored shapes: A is m×k or k×m, B is k×n or n×k
				let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
				let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
				let (lda, ldb, ccs) = (ar + 2, br + 1, m + 3);
				let a = rng.mat_c32(ar, ac, lda);
				let b = rng.mat_c32(br, bc, ldb);
				let c0 = rng.mat_c32(m, n, ccs);
				let (oa, lda2) = op_c32(&a, lda, ar, ac, ta);
				let (ob, ldb2) = op_c32(&b, ldb, br, bc, tb);
				for (alpha, beta) in [(C32::ONE, C32::ZERO), (C32::new(-0.7, 0.2), C32::new(0.4, 0.1)), (C32::ONE, C32::ONE)] {
					let mut c1 = c0.clone();
					let mut w1 = vec![C32::new(f32::NAN, f32::NAN); cgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)];
					cgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &oa, lda2, &ob, ldb2, beta, &mut c1, ccs, &mut w1).unwrap();
					let mut c2 = c0.clone();
					let mut w2 = vec![C32::new(f32::NAN, f32::NAN); cgemm_work(ta, tb, m, n, k)];
					cgemm(ta, tb, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c2, ccs, &mut w2).unwrap();
					for j in 0..n {
						for i in 0..m {
							assert!(
								bits_eq_cc(c1[j * ccs + i], c2[j * ccs + i]),
								"cgemm({ta:?},{tb:?}) vs explicit copy {m}x{k}x{n} ({i},{j})"
							);
						}
					}
				}
			}
		}
	}
}

#[test]
fn cgemm_packed_op_equals_the_copy_candidate_bitwise() {
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
				let a = rng.mat_c32(ar, ac, lda);
				let b = rng.mat_c32(br, bc, ldb);
				let c0 = rng.mat_c32(m, n, ccs);
				for (alpha, beta) in [(C32::ONE, C32::ZERO), (C32::new(-0.7, 0.2), C32::new(0.4, 0.1)), (C32::ONE, C32::ONE)] {
					let mut c1 = c0.clone();
					let mut w1 = vec![C32::new(f32::NAN, f32::NAN); cgemm_work(ta, tb, m, n, k)];
					cgemm(ta, tb, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c1, ccs, &mut w1).unwrap();
					// the routed wrapper may take the tiled shape below the threshold; the
					// packed shape is bit-identical to it, so the comparison still holds
					let mut c2 = c0.clone();
					let mut w2 = vec![C32::new(f32::NAN, f32::NAN); cgemm_packed_op_work(ta, tb, m, n, k)];
					cgemm_packed_op(ta, tb, alpha, m, n, k, &a, lda, &b, ldb, beta, &mut c2, ccs, &mut w2);
					for j in 0..n {
						for i in 0..m {
							assert!(
								bits_eq_cc(c1[j * ccs + i], c2[j * ccs + i]),
								"cgemm_packed_op({ta:?},{tb:?}) vs copy {m}x{k}x{n} ({i},{j})"
							);
						}
					}
				}
			}
		}
	}
	assert_eq!(cgemm_packed_op_work(Trans::NoTrans, Trans::NoTrans, 300, 40, 300), cgemm_packed_work(300, 40, 300));
}
