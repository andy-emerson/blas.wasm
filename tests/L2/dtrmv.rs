use super::common::*;
use blas_wasm::L2::*;

#[test]
fn trmv_bit_for_bit_all_variants() {
	let mut rng = Lcg(26);
	for &n in NS {
		for upper in [true, false] {
			for unit in [true, false] {
				let cs = n + 1;
				let a = rng.mat_f64(n, n, cs);
				let x0 = rng.vec_f64(n);
				let mut x = x0.clone();
				dtrmv(uplo(upper), Trans::NoTrans, diag(unit), n, &a, cs, &mut x).unwrap();
				// same-order scalar replay
				let mut xr = x0.clone();
				if upper {
					for j in 0..n {
						let t = xr[j];
						for i in 0..j {
							xr[i] += a[j * cs + i] * t;
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				} else {
					for j in (0..n).rev() {
						let t = xr[j];
						for i in j + 1..n {
							xr[i] += a[j * cs + i] * t;
						}
						if !unit {
							xr[j] = t * a[j * cs + j];
						}
					}
				}
				for i in 0..n {
					assert_eq!(
						x[i].to_bits(),
						xr[i].to_bits(),
						"dtrmv upper={upper} unit={unit} n={n} i={i}"
					);
				}
			}
		}
	}
}
