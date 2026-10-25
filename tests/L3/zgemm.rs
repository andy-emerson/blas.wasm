use super::common::*;
use blas_wasm::L3::*;
use blas_wasm::Trans;
use blas_wasm::C64;

#[test]
fn zgemm_bit_for_bit_and_bounded() {
	let mut rng = Lcg(131);
	for &(m, k, n) in DIMS {
		let (acs, bcs, ccs) = (m + 1, k + 2, m + 3);
		let a = rng.mat_c64(m, k, acs);
		let b = rng.mat_c64(k, n, bcs);
		let c0 = rng.mat_c64(m, n, ccs);
		for (alpha, beta) in [
			(C64::ONE, C64::ZERO),
			(C64::new(-0.7, 0.4), C64::new(0.4, -0.2)),
		] {
			let mut c = c0.clone();
			zgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &a, acs, &b, bcs, beta, &mut c, ccs, &mut vec![C64::ZERO; zgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
			// same-order scalar replay (zgemv per column = column-zaxpy)
			let mut cr = c0.clone();
			for j in 0..n {
				if beta == C64::ZERO {
					for i in 0..m {
						cr[j * ccs + i] = C64::ZERO;
					}
				} else if beta != C64::ONE {
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
						bits_eq_c(c[j * ccs + i], cr[j * ccs + i]),
						"zgemm bits {m}x{k}x{n} ({i},{j})"
					);
				}
			}
			// independent bound, different accumulation order
			for j in 0..n {
				for i in 0..m {
					let bt = if beta == C64::ZERO { C64::ZERO } else { beta * c0[j * ccs + i] };
					let want =
						alpha * comp_sum_c((0..k).map(|l| a[l * acs + i] * b[j * bcs + l])) + bt;
					let scale = comp_scale_c((0..k).map(|l| a[l * acs + i] * b[j * bcs + l]))
						* (alpha.abs1() + 1.0)
						+ bt.abs1();
					let tol = f64::EPSILON * (k.max(1) as f64) * 16.0 * scale + 1e-300;
					assert!(
						(c[j * ccs + i].re - want.re).abs() + (c[j * ccs + i].im - want.im).abs()
							<= tol,
						"zgemm bound ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn zgemm_col4_bit_identical_to_colaxpy() {
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
		let a = rng.mat_c64(m, k, acs);
		let b = rng.mat_c64(k, n, bcs);
		let c0 = rng.mat_c64(m, n, ccs);
		for (alpha, beta) in [
			(C64::ONE, C64::ZERO),
			(C64::new(-0.7, 0.2), C64::new(0.4, 0.1)),
			(C64::new(0.3, -0.6), C64::ONE),
		] {
			let mut c1 = c0.clone();
			zgemm_colaxpy(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut c2 = c0.clone();
			zgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &a, acs, &b, bcs, beta, &mut c2, ccs, &mut vec![C64::ZERO; zgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
			for j in 0..n {
				for i in 0..m {
					assert!(
						bits_eq_c(c1[j * ccs + i], c2[j * ccs + i]),
						"zgemm col4 vs colaxpy {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn zgemm_short_storage_is_an_error() {
	let a = [C64::ONE; 4];
	let b = [C64::ONE; 3];
	assert_eq!(zgemm(Trans::NoTrans, Trans::NoTrans, 2, 2, 2, C64::ONE, &a, 2, &b, 2, C64::ZERO, &mut [C64::ZERO; 4], 2, &mut vec![C64::ZERO; zgemm_work(Trans::NoTrans, Trans::NoTrans, 2, 2, 2)]), Err(blas_wasm::Error { routine: "zgemm", arg: 9, kind: blas_wasm::ErrorKind::Length }));
}

#[test]
fn zgemm_packed_bit_identical_to_colaxpy() {
	let mut rng = Lcg(141);
	// sizes crossing the packed-path boundaries: KC (256) exact / +1 /
	// with remainder, MC (64) row-blocking, MR=2 row tails, column tails
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
		let a = rng.mat_c64(m, k, acs);
		let b = rng.mat_c64(k, n, bcs);
		let c0 = rng.mat_c64(m, n, ccs);
		for (alpha, beta) in [
			(C64::ONE, C64::ZERO),
			(C64::new(-0.7, 0.2), C64::new(0.4, 0.1)),
			(C64::new(0.3, -0.6), C64::ONE),
		] {
			let mut c1 = c0.clone();
			zgemm_colaxpy(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c1, ccs);
			let mut c2 = c0.clone();
			// a NaN-filled workspace of exactly the queried size: any read of
			// scratch the packer did not write would poison C
			let mut work = vec![C64::new(f64::NAN, f64::NAN); zgemm_packed_work(m, n, k)];
			zgemm_packed(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c2, ccs, &mut work);
			for j in 0..n {
				for i in 0..m {
					assert_eq!(
						c1[j * ccs + i].re.to_bits(),
						c2[j * ccs + i].re.to_bits(),
						"packed vs column-zaxpy re {m}x{k}x{n} ({i},{j})"
					);
					assert_eq!(
						c1[j * ccs + i].im.to_bits(),
						c2[j * ccs + i].im.to_bits(),
						"packed vs column-zaxpy im {m}x{k}x{n} ({i},{j})"
					);
				}
			}
		}
	}
}

#[test]
fn zgemm_dispatch_packed_zone_bit_identical() {
	// A = 260x256x16B = 1.06 MB >= the 1 MB packed threshold — the
	// dispatcher routes packed; replay against colaxpy
	let mut rng = Lcg(143);
	let (m, k, n) = (260, 256, 8);
	let (acs, bcs, ccs) = (m, k, m);
	let a = rng.mat_c64(m, k, acs);
	let b = rng.mat_c64(k, n, bcs);
	let c0 = rng.mat_c64(m, n, ccs);
	let (alpha, beta) = (C64::new(-0.7, 0.2), C64::new(0.4, 0.1));
	let mut c1 = c0.clone();
	zgemm_colaxpy(alpha, m, n, k, &a, acs, &b, bcs, beta, &mut c1, ccs);
	let mut c2 = c0.clone();
	zgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &a, acs, &b, bcs, beta, &mut c2, ccs, &mut vec![C64::ZERO; zgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
	for i in 0..c1.len() {
		assert_eq!(c1[i].re.to_bits(), c2[i].re.to_bits(), "dispatch(packed) re @{i}");
		assert_eq!(c1[i].im.to_bits(), c2[i].im.to_bits(), "dispatch(packed) im @{i}");
	}
}

/// The explicit copy the flags are defined against: `op(x)` column-major
/// at a leading dimension of rows+1 (deliberately not the tight one).
fn op_z(x: &[C64], ld: usize, rows: usize, cols: usize, t: Trans) -> (Vec<C64>, usize) {
	let (r, c) = match t {
		Trans::NoTrans => (rows, cols),
		_ => (cols, rows),
	};
	let ldo = r + 1;
	let mut o = vec![C64::ZERO; ldo * c];
	for j in 0..c {
		for i in 0..r {
			o[j * ldo + i] = match t {
				Trans::NoTrans => x[j * ld + i],
				Trans::Trans => x[i * ld + j],
				Trans::ConjTrans => x[i * ld + j].conj(),
			};
		}
	}
	(o, ldo)
}

#[test]
fn zgemm_flags_equal_the_explicit_transpose_bitwise() {
	// the definition of the flags: op(A)·op(B) through transa/transb is
	// bit-identical to the no-transpose call on explicitly transposed
	// (and conjugated) copies, for every flag pair, below and above the
	// packed threshold, with the workspace NaN-filled at the queried size
	let mut rng = Lcg(0x5EED_5EED);
	let flags = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
	for (m, n, k) in [(1, 1, 1), (5, 7, 3), (16, 16, 16), (33, 9, 70), (64, 65, 63), (256, 8, 300)] {
		for ta in flags {
			for tb in flags {
				// stored shapes: A is m×k or k×m, B is k×n or n×k
				let (ar, ac) = if ta == Trans::NoTrans { (m, k) } else { (k, m) };
				let (br, bc) = if tb == Trans::NoTrans { (k, n) } else { (n, k) };
				let (lda, ldb, ccs) = (ar + 2, br + 1, m + 3);
				let a = rng.mat_c64(ar, ac, lda);
				let b = rng.mat_c64(br, bc, ldb);
				let c0 = rng.mat_c64(m, n, ccs);
				let (oa, lda2) = op_z(&a, lda, ar, ac, ta);
				let (ob, ldb2) = op_z(&b, ldb, br, bc, tb);
				for (alpha, beta) in [(C64::ONE, C64::ZERO), (C64::new(-0.7, 0.2), C64::new(0.4, 0.1)), (C64::ONE, C64::ONE)] {
					let mut c1 = c0.clone();
					let mut w1 = vec![C64::new(f64::NAN, f64::NAN); zgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)];
					zgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, &oa, lda2, &ob, ldb2, beta, &mut c1, ccs, &mut w1).unwrap();
					let mut c2 = c0.clone();
					let mut w2 = vec![C64::new(f64::NAN, f64::NAN); zgemm_work(ta, tb, m, n, k)];
					zgemm(ta, tb, m, n, k, alpha, &a, lda, &b, ldb, beta, &mut c2, ccs, &mut w2).unwrap();
					for j in 0..n {
						for i in 0..m {
							assert!(
								bits_eq_c(c1[j * ccs + i], c2[j * ccs + i]),
								"zgemm({ta:?},{tb:?}) vs explicit copy {m}x{k}x{n} ({i},{j})"
							);
						}
					}
				}
			}
		}
	}
}
